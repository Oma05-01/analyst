"""Local experimental UI for manually capturing projected or official Premier League starting XIs.

Run from repository root:
    streamlit run src/lineup_entry_app.py

Reads upcoming fixtures from PostgreSQL and the current-season FPL roster CSVs.
Writes append-only projected/official lineup snapshots to data/lineups/confirmed_lineups.jsonl.
It does NOT write to PostgreSQL, make predictions, or alter locked forecasts.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import subprocess
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FPL_DATA_PATH = (
    r"C:\Users\user\Downloads\Fantasy-Premier-League-master"
    r"\Fantasy-Premier-League-master\data"
)
FPL_DATA_PATH = Path(os.getenv("FPL_DATA_PATH", DEFAULT_FPL_DATA_PATH)).expanduser()
FPL_SEASON = os.getenv("FPL_SEASON", "2026-27")
LINEUP_STORE = Path(os.getenv("LINEUP_STORE", "data/lineups/confirmed_lineups.jsonl"))
UPCOMING_STATUSES = ("SCHEDULED", "TIMED")
POSITION_LABELS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Aliases are used only to suggest a default FPL-team mapping. The user can change it.
TEAM_ALIASES = {
    "afc bournemouth": ["bournemouth", "afc bournemouth"],
    "brighton and hove albion": ["brighton", "brighton and hove albion", "brighton hove albion"],
    "manchester city": ["man city", "manchester city"],
    "manchester united": ["man utd", "man united", "manchester united"],
    "newcastle united": ["newcastle", "newcastle united"],
    "nottingham forest": ["nottingham forest", "nottm forest", "nott'm forest", "forest"],
    "tottenham hotspur": ["tottenham", "tottenham hotspur", "spurs"],
    "west ham united": ["west ham", "west ham united"],
    "wolverhampton wanderers": ["wolves", "wolverhampton", "wolverhampton wanderers"],
    "ipswich town": ["ipswich", "ipswich town"],
    "coventry city": ["coventry", "coventry city"],
    "hull city": ["hull", "hull city"],
    "leeds united": ["leeds", "leeds united"],
    "leicester city": ["leicester", "leicester city"],
    "sheffield united": ["sheffield utd", "sheffield united"],
    "sheffield wednesday": ["sheffield wednesday"],
    "queens park rangers": ["qpr", "queens park rangers"],
    "sunderland afc": ["sunderland", "sunderland afc"],
}


def clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalise_team(value: Any) -> str:
    text_value = clean_string(value).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text_value)


def team_match_keys(name: str) -> set[str]:
    n = normalise_team(name)
    keys = {n}
    # Get aliases by matching either a canonical name or one of its aliases.
    for canonical, aliases in TEAM_ALIASES.items():
        group = [canonical, *aliases]
        if n in {normalise_team(item) for item in group}:
            keys.update(normalise_team(item) for item in group)
    return keys


def suggested_fpl_team(fixture_team: str, team_names: list[str]) -> str | None:
    target_keys = team_match_keys(fixture_team)
    matches = [name for name in team_names if target_keys.intersection(team_match_keys(name))]
    return matches[0] if len(matches) == 1 else None


@st.cache_resource
def db_engine() -> Engine:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing. Check the project's .env file.")
    return create_engine(database_url, pool_pre_ping=True, pool_size=1, max_overflow=1)


@st.cache_data(ttl=30)
def load_upcoming_fixtures() -> pd.DataFrame:
    query = text("""
        SELECT m.id AS match_id, m.season, m.matchweek, m.kickoff_utc,
               ht.name AS home_team, awt.name AS away_team, m.status
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams awt ON awt.id = m.away_team_id
        WHERE m.status = ANY(:statuses)
          AND m.home_goals IS NULL AND m.away_goals IS NULL
          AND m.kickoff_utc > NOW()
        ORDER BY m.kickoff_utc, m.id
    """)
    with db_engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"statuses": list(UPCOMING_STATUSES)})
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    return df


def load_rosters() -> tuple[pd.DataFrame, pd.DataFrame]:
    season_dir = FPL_DATA_PATH / FPL_SEASON
    teams_path = season_dir / "teams.csv"
    players_path = season_dir / "players_raw.csv"
    missing = [str(path) for path in (teams_path, players_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Could not find the current-season FPL roster files:\n- " + "\n- ".join(missing)
            + "\nSet FPL_DATA_PATH and/or FPL_SEASON in .env to the correct archive location."
        )

    teams = pd.read_csv(teams_path)
    players = pd.read_csv(players_path)
    required_team_cols = {"id", "name"}
    required_player_cols = {"id", "team", "element_type"}
    if not required_team_cols.issubset(teams.columns):
        raise ValueError(f"teams.csv must contain columns: {sorted(required_team_cols)}")
    if not required_player_cols.issubset(players.columns):
        raise ValueError(f"players_raw.csv must contain columns: {sorted(required_player_cols)}")

    teams = teams[["id", "name"]].copy()
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce")
    teams["name"] = teams["name"].astype(str).str.strip()
    teams = teams.dropna(subset=["id", "name"]).drop_duplicates("id")
    teams["id"] = teams["id"].astype(int)

    players = players.copy()
    for col in ("id", "team", "element_type"):
        players[col] = pd.to_numeric(players[col], errors="coerce")
    players = players.dropna(subset=["id", "team", "element_type"])
    players = players[players["element_type"].isin(POSITION_LABELS)].copy()
    players["id"] = players["id"].astype(int)
    players["team"] = players["team"].astype(int)
    players["position"] = players["element_type"].map(POSITION_LABELS)

    first = players["first_name"].fillna("").astype(str).str.strip() if "first_name" in players else pd.Series("", index=players.index)
    second = players["second_name"].fillna("").astype(str).str.strip() if "second_name" in players else pd.Series("", index=players.index)
    full = (first + " " + second).str.strip()
    if "web_name" in players:
        web_name = players["web_name"].fillna("").astype(str).str.strip()
        full = full.where(full.ne(""), web_name)
    players["player_name"] = full.where(full.ne(""), "Player " + players["id"].astype(str))
    players["label"] = players["player_name"] + " (" + players["position"] + ")"
    if "code" in players:
        players["player_code"] = players["code"].where(players["code"].notna(), None)
    else:
        players["player_code"] = None
    return teams, players


def read_confirmed_records() -> list[dict[str, Any]]:
    if not LINEUP_STORE.exists():
        return []
    records: list[dict[str, Any]] = []
    with LINEUP_STORE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("match_id") is not None:
                    records.append(row)
            except json.JSONDecodeError:
                st.warning(f"Ignoring invalid JSON on line {line_number} of {LINEUP_STORE}.")
    return records


def record_lineup_type(record: dict[str, Any]) -> str:
    """Read v2 type; treat legacy explicitly checked snapshots as official."""
    value = record.get("lineup_type")
    if value in {"projected", "official_lineup"}:
        return value
    if record.get("confirmation_check") is True:
        return "official_lineup"
    return "projected"


def latest_record_for_match(records: list[dict[str, Any]], match_id: int,
                            lineup_type: str | None = None) -> dict[str, Any] | None:
    candidates = [r for r in records if int(r.get("match_id", -1)) == int(match_id)]
    if lineup_type is not None:
        candidates = [r for r in candidates if record_lineup_type(r) == lineup_type]
    if not candidates:
        return None
    return max(candidates, key=lambda r: int(r.get("revision", 0)))


def ids_for_position(team_players: pd.DataFrame, position: str) -> list[int]:
    subset = team_players.loc[team_players["position"].eq(position), "id"]
    return sorted(int(value) for value in subset.tolist())


def player_label_map(team_players: pd.DataFrame) -> dict[int, str]:
    return {int(row.id): str(row.label) for row in team_players.itertuples(index=False)}


def existing_ids(existing_lineup: list[dict[str, Any]] | None, position: str) -> list[int]:
    if not existing_lineup:
        return []
    result = []
    for player in existing_lineup:
        try:
            if player.get("position") == position:
                result.append(int(player["element_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def render_xi(
    team_players: pd.DataFrame,
    label: str,
    key_prefix: str,
    existing_lineup: list[dict[str, Any]] | None,
) -> tuple[list[int], dict[int, str]]:
    st.subheader(label)

    labels = player_label_map(team_players)
    indexed = team_players.set_index("id", drop=False)

    existing_by_id = {}
    for player in existing_lineup or []:
        try:
            existing_by_id[int(player["element_id"])] = player
        except (KeyError, TypeError, ValueError):
            continue

    # Keep the goalkeeper slot tied to the registered goalkeeper roster.
    gk_options = ids_for_position(team_players, "GKP")
    if not gk_options:
        st.error(f"No goalkeepers were found in the selected FPL roster for {label}.")
        return [], {}

    old_gk = existing_ids(existing_lineup, "GKP")
    gk_default = old_gk[0] if old_gk and old_gk[0] in gk_options else gk_options[0]

    gk = st.selectbox(
        "Goalkeeper (1)",
        gk_options,
        index=gk_options.index(gk_default),
        format_func=lambda pid: labels.get(int(pid), str(pid)),
        key=f"{key_prefix}_gkp",
    )

    # All registered outfield players are eligible, regardless of FPL position.
    outfield_players = team_players.loc[
        team_players["position"].ne("GKP")
    ]
    outfield_options = sorted(int(pid) for pid in outfield_players["id"])

    old_outfield = []
    for player in existing_lineup or []:
        try:
            pid = int(player["element_id"])
            if pid in outfield_options:
                old_outfield.append(pid)
        except (KeyError, TypeError, ValueError):
            continue

    selected_outfield = st.multiselect(
        "Outfield players (choose 10)",
        outfield_options,
        default=old_outfield[:10],
        max_selections=10,
        format_func=lambda pid: labels.get(int(pid), str(pid)),
        key=f"{key_prefix}_outfield",
    )

    # Assign each selected player a match-specific tactical role.
    role_options = ["DEF", "MID", "FWD"]
    playing_positions: dict[int, str] = {int(gk): "GKP"}

    if selected_outfield:
        st.markdown("**Assign playing positions**")
        st.caption(
            "The position in brackets is the player's FPL registration. "
            "Choose the role they are expected to play in this match."
        )

        for pid in selected_outfield:
            pid = int(pid)
            player = indexed.loc[pid]
            fpl_position = str(player["position"])

            old_player = existing_by_id.get(pid, {})
            old_role = old_player.get("playing_position") or old_player.get("position")

            if old_role not in role_options:
                old_role = fpl_position if fpl_position in role_options else "MID"

            role = st.selectbox(
                f"{player['player_name']} ({fpl_position})",
                role_options,
                index=role_options.index(old_role),
                key=f"{key_prefix}_role_{pid}",
            )
            playing_positions[pid] = role

    selected_ids = [int(gk)] + [int(pid) for pid in selected_outfield]

    role_counts = {
        role: sum(1 for value in playing_positions.values() if value == role)
        for role in ["DEF", "MID", "FWD"]
    }

    st.caption(
        f"Selected players: {len(selected_ids)} / 11 | "
        f"DEF {role_counts['DEF']} · "
        f"MID {role_counts['MID']} · "
        f"FWD {role_counts['FWD']}"
    )

    return selected_ids, playing_positions

def serialise_lineup(
    selected_ids: list[int],
    team_players: pd.DataFrame,
    playing_positions: dict[int, str],
) -> list[dict[str, Any]]:
    indexed = team_players.set_index("id", drop=False)
    result = []

    for pid in selected_ids:
        pid = int(pid)
        row = indexed.loc[pid]

        code = row.get("player_code")
        if pd.isna(code):
            code = None
        else:
            try:
                code = int(code)
            except (TypeError, ValueError):
                code = str(code)

        fpl_position = str(row["position"])

        result.append({
            "element_id": pid,
            "player_code": code,
            "player_name": str(row["player_name"]),

            # Retain the original field for compatibility.
            "position": fpl_position,

            # Explicitly preserve registration and match role separately.
            "fpl_position": fpl_position,
            "playing_position": playing_positions[pid],
        })

    return result

def save_append_only(record: dict[str, Any]) -> None:
    LINEUP_STORE.parent.mkdir(parents=True, exist_ok=True)
    with LINEUP_STORE.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


st.set_page_config(page_title="Manual Lineup Capture", page_icon="⚽", layout="wide")
st.title("Experimental Starting XI Capture")
st.caption("Private local tool · projected and official lineup snapshots · does not generate or publish predictions")
st.warning(
    "Keep this app local. It reads upcoming fixtures from PostgreSQL but writes lineup snapshots only "
    "to a local JSONL file. A projected XI is not an official lineup. It does not update locked predictions or the public API."
)

with st.expander("Configuration", expanded=False):
    st.write(f"FPL data directory: `{FPL_DATA_PATH}`")
    st.write(f"FPL season folder: `{FPL_SEASON}`")
    st.write(f"Lineup snapshot file: `{LINEUP_STORE}`")
    st.write("Override `FPL_DATA_PATH`, `FPL_SEASON`, or `LINEUP_STORE` in your `.env` if needed.")

try:
    fixtures = load_upcoming_fixtures()
except Exception as exc:
    st.error(f"Could not load upcoming fixtures from PostgreSQL: {exc}")
    st.stop()

try:
    teams, players = load_rosters()
except Exception as exc:
    st.error(f"Could not load the current-season FPL roster: {exc}")
    st.stop()

if fixtures.empty:
    st.info("No future SCHEDULED or TIMED fixtures were found in the database.")
    st.stop()

fpl_team_names = sorted(teams["name"].astype(str).tolist())
fpl_team_id_by_name = {str(row["name"]): int(row["id"]) for _, row in teams.iterrows()}

fixture_options = fixtures.to_dict("records")
fixture_labels = {
    int(row["match_id"]): (
        f"{pd.Timestamp(row['kickoff_utc']).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"{row['home_team']} vs {row['away_team']} · Match {row['match_id']}"
    ) for row in fixture_options
}
selected_match_id = st.selectbox(
    "Upcoming fixture", [int(row["match_id"]) for row in fixture_options],
    format_func=lambda match_id: fixture_labels[int(match_id)],
)
fixture = next(row for row in fixture_options if int(row["match_id"]) == int(selected_match_id))

lineup_type = st.radio(
    "Lineup status",
    options=["projected", "official_lineup"],
    format_func=lambda value: "Projected XI (your estimate)" if value == "projected" else "Official starting XI (published team sheet)",
    horizontal=True,
    help="Use Projected XI before club team sheets are released. Choose Official only after checking the published starting lineups.",
    key=f"lineup_type_{selected_match_id}",
)

try:
    records = read_confirmed_records()
except OSError as exc:
    st.error(f"Could not read existing lineup snapshots: {exc}")
    st.stop()
existing_record = latest_record_for_match(records, int(selected_match_id), lineup_type)
current_revision = int(existing_record.get("revision", 0)) if existing_record else 0

st.markdown(
    f"**Fixture:** {fixture['home_team']} vs {fixture['away_team']}  \n"
    f"**Kickoff:** {utc_iso(fixture['kickoff_utc'])}  ·  "
    f"**Status:** {fixture['status']}  ·  **Match ID:** {fixture['match_id']}"
)
if existing_record:
    st.info(
        f"A {lineup_type.replace('_', ' ')} snapshot already exists (revision {current_revision}, "
        f"saved {existing_record.get('confirmed_at_utc', 'time unknown')}). "
        "Saving again will append a new revision of this type and preserve all previous records."
    )
else:
    st.info(f"No {lineup_type.replace('_', ' ')} snapshot exists yet for this fixture.")

home_suggest = suggested_fpl_team(
    str(fixture["home_team"]),
    fpl_team_names,
)
away_suggest = suggested_fpl_team(
    str(fixture["away_team"]),
    fpl_team_names,
)

# Stop instead of guessing if either roster cannot be matched.
if home_suggest is None:
    st.error(
        f"Could not confidently match '{fixture['home_team']}' "
        "to an FPL roster. Check TEAM_ALIASES in "
        "lineup_entry_app.py."
    )
    st.stop()

if away_suggest is None:
    st.error(
        f"Could not confidently match '{fixture['away_team']}' "
        "to an FPL roster. Check TEAM_ALIASES in "
        "lineup_entry_app.py."
    )
    st.stop()

# Enforce the automatically matched rosters.
# No manual roster dropdowns, so the teams cannot be accidentally swapped.
home_fpl_name = home_suggest
away_fpl_name = away_suggest

st.caption(
    f"Home: {fixture['home_team']} | FPL roster: {home_fpl_name}"
)
st.caption(
    f"Away: {fixture['away_team']} | FPL roster: {away_fpl_name}"
)

if normalise_team(home_fpl_name) == normalise_team(away_fpl_name):
    st.error("Home and away sides cannot use the same FPL team roster.")
    st.stop()

home_fpl_id = fpl_team_id_by_name[home_fpl_name]
away_fpl_id = fpl_team_id_by_name[away_fpl_name]
home_players = players.loc[players["team"].eq(home_fpl_id)].copy()
away_players = players.loc[players["team"].eq(away_fpl_id)].copy()
if home_players.empty or away_players.empty:
    st.error("One of the selected FPL teams has no players in players_raw.csv.")
    st.stop()

old_home = existing_record.get("home_lineup", []) if existing_record else []
old_away = existing_record.get("away_lineup", []) if existing_record else []
key_suffix = f"match_{selected_match_id}_{lineup_type}_rev_{current_revision}_{home_fpl_id}_{away_fpl_id}"

home_col, away_col = st.columns(2, gap="large")
with home_col:
    st.markdown(f"### Home: {fixture['home_team']}")
    st.caption(f"Using FPL roster: {home_fpl_name}")
    home_selected, home_roles = render_xi(
    home_players,
    str(fixture["home_team"]),
    f"home_{key_suffix}",
    old_home,
)
with away_col:
    st.markdown(f"### Away: {fixture['away_team']}")
    st.caption(f"Using FPL roster: {away_fpl_name}")
    away_selected, away_roles = render_xi(away_players, str(fixture["away_team"]), f"away_{key_suffix}", old_away)

notes = st.text_area("Notes (optional; e.g. source or correction reason)", key=f"notes_{key_suffix}")
if lineup_type == "official_lineup":
    acknowledged = st.checkbox(
        "I have checked both selections against the officially published starting XIs.",
        key=f"confirm_official_{key_suffix}",
    )
else:
    acknowledged = st.checkbox(
        "I understand these are my projected XIs, not officially published lineups.",
        key=f"ack_projected_{key_suffix}",
    )


def valid_xi(selected: list[int]) -> tuple[bool, str]:
    if len(selected) != 11:
        return False, f"Select exactly 11 players; currently selected {len(selected)}."
    if len(set(selected)) != 11:
        return False, "A player is selected more than once."
    return True, "Valid XI"

home_valid, home_message = valid_xi(home_selected)
away_valid, away_message = valid_xi(away_selected)
status_col1, status_col2 = st.columns(2)
with status_col1:
    st.write(("✅ " if home_valid else "⚠️ ") + f"Home: {home_message}")
with status_col2:
    st.write(("✅ " if away_valid else "⚠️ ") + f"Away: {away_message}")

if st.button(
    "Confirm and save lineup snapshot",
    type="primary",
    disabled=not (home_valid and away_valid and acknowledged),
):
    # Re-read before appending to avoid reusing a stale revision number.
    fresh_records = read_confirmed_records()
    latest = latest_record_for_match(
        fresh_records,
        int(selected_match_id),
        lineup_type,
    )
    latest_revision = int(latest.get("revision", 0)) if latest else 0

    if latest_revision != current_revision:
        st.error(
            "A newer lineup revision was saved after this page loaded. "
            "Refresh the page before saving."
        )

    else:
        revision = current_revision + 1

        record = {
            "schema_version": 2,
            "lineup_type": lineup_type,
            "match_id": int(fixture["match_id"]),
            "season": str(fixture["season"]),
            "matchweek": (
                None if pd.isna(fixture["matchweek"])
                else int(fixture["matchweek"])
            ),
            "kickoff_utc": utc_iso(fixture["kickoff_utc"]),

            # Fixture and roster metadata
            "home_team": str(fixture["home_team"]),
            "away_team": str(fixture["away_team"]),
            "fixture_status_at_confirmation": str(fixture["status"]),
            "fpl_season": FPL_SEASON,
            "home_fpl_team": {
                "id": int(home_fpl_id),
                "name": str(home_fpl_name),
            },
            "away_fpl_team": {
                "id": int(away_fpl_id),
                "name": str(away_fpl_name),
            },

            # Starting XIs, including match-specific playing positions
            "home_lineup": serialise_lineup(
                home_selected, home_players, home_roles
            ),
            "away_lineup": serialise_lineup(
                away_selected, away_players, away_roles
            ),

            "lineup_source": (
                "manual_projected_xi"
                if lineup_type == "projected"
                else "manual_official_starting_xi"
            ),
            "revision": revision,
            "supersedes_revision": (
                current_revision if current_revision else None
            ),
            "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
            "confirmation_check": (
                bool(acknowledged)
                if lineup_type == "official_lineup"
                else False
            ),
            "user_acknowledged_lineup_type": bool(acknowledged),
            "notes": notes.strip() or None,
            "forecast_status": "not_generated",
        }

        try:
            save_append_only(record)

        except OSError as exc:
            st.error(f"Could not save the lineup snapshot: {exc}")

        else:
            existing_record = record
            current_revision = revision

            st.success(
                f"Saved {lineup_type.replace('_', ' ')} "
                f"snapshot revision {revision}."
            )

            # Show the metadata as well, so you can verify it was saved.
            st.json({
                "match_id": record["match_id"],
                "lineup_type": record["lineup_type"],
                "revision": record["revision"],
                "home_team": record["home_team"],
                "away_team": record["away_team"],
                "home_fpl_team": record["home_fpl_team"],
                "away_fpl_team": record["away_fpl_team"],
                "confirmed_at_utc": record["confirmed_at_utc"],
                "home_starters": len(record["home_lineup"]),
                "away_starters": len(record["away_lineup"]),
            })

            # Generate the experimental forecast only after a successful save.
            command = [
                sys.executable,
                "-m",
                "src.generate_context_lineup_forecast",
                "--match-id",
                str(record["match_id"]),
                "--lineup-type",
                lineup_type,
                "--save",
            ]

            try:
                with st.spinner(
                    "Lineup saved. Generating the experimental forecast..."
                ):
                    result = subprocess.run(
                        command,
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=False,
                    )

            except subprocess.TimeoutExpired:
                st.error(
                    "The lineup was saved, but forecast generation timed out. "
                    "You can rerun the generator manually."
                )

            except OSError as exc:
                st.error(
                    "The lineup was saved, but the forecast generator "
                    f"could not be started: {exc}"
                )

            else:
                output = "\n".join(
                    part
                    for part in (result.stdout, result.stderr)
                    if part
                ).strip()

                if result.returncode == 0:
                    st.success(
                        "Experimental forecast generated or already up to date. "
                        "Refresh Premier Quant to view it."
                    )
                else:
                    st.error(
                        "The lineup was saved, but forecast generation failed. "
                        "Your existing locked predictions remain unchanged."
                    )

                if output:
                    with st.expander("Forecast generator output"):
                        st.code(output[-12000:], language="text")

with st.expander(f"Latest saved {lineup_type.replace(chr(95), chr(32))} snapshot details", expanded=False):
    if existing_record:
        st.json({
            "match_id": existing_record.get("match_id"),
            "lineup_type": record_lineup_type(existing_record),
            "revision": existing_record.get("revision"),
            "confirmed_at_utc": existing_record.get("confirmed_at_utc"),
            "home_lineup": existing_record.get("home_lineup", []),
            "away_lineup": existing_record.get("away_lineup", []),
            "forecast_status": existing_record.get("forecast_status"),
            "notes": existing_record.get("notes"),
        })
    else:
        st.write("No snapshot yet.")
