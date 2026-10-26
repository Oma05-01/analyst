#!/usr/bin/env python3
"""Sync completed Premier League results into PostgreSQL, preserving prediction rows.

The default mode is DRY-RUN. It does not write to PostgreSQL.

Football-Data.co.uk source files use the standard E0 columns such as Date, Time,
HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HC, AC, HY, AY, HR, AR and Referee.
The Date/Time fields are local UK time and are converted with Europe/London rules.

Examples (run from repository root):
    python -m src.sync_completed_results --download
    # Inspect output, take a database backup, then:
    python -m src.sync_completed_results --download --apply

Or supply already-downloaded files:
    python -m src.sync_completed_results \
        --season-2526 data/raw/E0_2526.csv \
        --season-2627 data/raw/E0_2627.csv

Only rows with both full-time goals are ingested. Unplayed / incomplete rows in
2026-27 are ignored. The completed 2025-26 season is required to have 380 results.
Writes use the existing (season, home_team_id, away_team_id) uniqueness key. For
an existing fixture with a saved prediction, the stored kickoff timestamp is kept
so a result import cannot alter the prediction's measured lead time. Predictions
and locked prediction payloads themselves are never updated.

Credit the source in your README and review the source's reuse terms before
redistributing source data.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from src.db import get_db_connection

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_2526 = RAW_DIR / "E0_2526.csv"
DEFAULT_2627 = RAW_DIR / "E0_2627.csv"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc

# football-data.co.uk short labels that do not always exactly match canonical
# team names. These are aliases only; every target must still resolve against the
# project's `teams` table before any writes happen.
SOURCE_TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "newcastle": "newcastle united",
    "nottm forest": "nottingham forest",
    "notts forest": "nottingham forest",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "west ham": "west ham united",
    "brighton": "brighton and hove albion",
    "leeds": "leeds united",
    "luton": "luton town",
    "ipswich": "ipswich town",
    "sheffield utd": "sheffield united",
    "sheffield weds": "sheffield wednesday",
    "west brom": "west bromwich albion",
    "wba": "west bromwich albion",
    "norwich": "norwich city",
    "palace": "crystal palace",
    "bournemouth": "bournemouth",
    "coventry": "coventry city",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def team_key(value: object) -> str:
    """Normalize names conservatively; don't fuzzy-match unknown clubs."""
    if value is None or pd.isna(value):
        return ""
    name = str(value).strip().casefold()
    name = name.replace("&", " and ")
    name = name.replace("'", "").replace(".", "")
    name = re.sub(r"^afc\s+", "", name)
    name = re.sub(r"\s+fc$", "", name)
    name = re.sub(r"\s+afc$", "", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def clean_optional_int(value: object, field: str, row_label: str) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or float(number) % 1 != 0 or float(number) < 0:
        raise ValueError(f"Invalid {field}={value!r} in {row_label}")
    return int(number)


def local_date_time_to_utc(date_value: object, time_value: object, row_label: str) -> datetime:
    date = pd.to_datetime(date_value, dayfirst=True, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"Invalid/missing Date in {row_label}: {date_value!r}")
    time_text = "" if time_value is None or pd.isna(time_value) else str(time_value).strip()
    if not time_text or time_text.lower() in {"nan", "nat", "none"}:
        raise ValueError(f"Missing Time in {row_label}; refusing to invent a kickoff time")
    parsed_time = None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(time_text, fmt).time()
            break
        except ValueError:
            continue
    if parsed_time is None:
        raise ValueError(f"Unsupported Time={time_value!r} in {row_label}")
    local_naive = datetime.combine(date.date(), parsed_time)
    return local_naive.replace(tzinfo=LONDON).astimezone(UTC)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source file: {path}\n"
            "Run with --download, or place the E0 season CSV at this path."
        )
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [str(c).strip().casefold() for c in df.columns]
    required = {"date", "time", "hometeam", "awayteam", "fthg", "ftag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is not a standard Football-Data E0 CSV. Missing columns: "
            f"{sorted(missing)}. Found: {list(df.columns)}"
        )
    if "div" in df.columns:
        divisions = set(df["div"].dropna().astype(str).str.strip().str.upper())
        if divisions and divisions != {"E0"}:
            raise ValueError(f"{path} contains unexpected division values: {sorted(divisions)}")
    return df


def download_csv(code: str, destination: Path) -> None:
    url = BASE_URL.format(code=code)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "football-analyst-agent/1.0 (research data import)"})
    logging.info("Downloading %s", url)
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    sample = payload[:300].lstrip().lower()
    if not payload or sample.startswith(b"<!doctype html") or sample.startswith(b"<html"):
        raise ValueError(f"The source URL did not return a CSV file: {url}")
    destination.write_bytes(payload)
    logging.info("Saved %s (%s bytes)", destination, len(payload))


def load_team_mappings(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, aliases FROM teams")
        rows = cur.fetchall()
    mapping: dict[str, int] = {}

    def register(label: object, team_id: int) -> None:
        key = team_key(label)
        if not key:
            return
        old = mapping.get(key)
        if old is not None and old != int(team_id):
            raise ValueError(f"Ambiguous database team alias {label!r}: IDs {old} and {team_id}")
        mapping[key] = int(team_id)

    for team_id, name, aliases in rows:
        register(name, team_id)
        if aliases:
            parsed = json.loads(aliases) if isinstance(aliases, str) else aliases
            if isinstance(parsed, (list, tuple)):
                for alias in parsed:
                    register(alias, team_id)
            elif isinstance(parsed, dict):
                for alias in parsed.keys():
                    register(alias, team_id)
            else:
                raise ValueError(f"Unsupported aliases value for team {name!r}: {type(parsed).__name__}")

    return mapping


def resolve_team(raw_name: object, team_map: dict[str, int]) -> int | None:
    source_key = team_key(raw_name)
    candidates = [source_key]
    canonical_key = SOURCE_TEAM_ALIASES.get(source_key)
    if canonical_key and canonical_key not in candidates:
        candidates.append(canonical_key)
    found = {team_map[key] for key in candidates if key in team_map}
    if len(found) > 1:
        raise ValueError(f"Source team name {raw_name!r} resolves to multiple database team IDs: {sorted(found)}")
    return next(iter(found)) if found else None


def _read_season(path: Path, season: str, require_full_season: bool) -> list[dict]:
    df = read_csv(path)
    home_present = df["fthg"].notna()
    away_present = df["ftag"].notna()
    partial = home_present ^ away_present
    if partial.any():
        examples = df.loc[partial, ["date", "hometeam", "awayteam", "fthg", "ftag"]].head(5)
        raise ValueError(f"Rows have only one full-time score in {path}; refusing partial results:\n{examples}")

    finished = df[home_present & away_present].copy()
    if finished.empty:
        raise ValueError(f"No completed matches with both scores found in {path}")
    if require_full_season and len(finished) != 380:
        raise ValueError(f"Expected 380 completed Premier League matches in {path}; found {len(finished)}. No writes made.")
    if finished[["hometeam", "awayteam"]].isna().any().any():
        raise ValueError(f"Missing home/away team names in completed rows of {path}")

    for col in ("fthg", "ftag"):
        values = pd.to_numeric(finished[col], errors="coerce")
        invalid = values.isna() | (values < 0) | (values % 1 != 0)
        if invalid.any():
            raise ValueError(f"Invalid full-time scores in {col} of {path}")
        finished[col] = values.astype(int)

    finished["_home_key"] = finished["hometeam"].map(team_key)
    finished["_away_key"] = finished["awayteam"].map(team_key)
    if (finished["_home_key"] == finished["_away_key"]).any():
        raise ValueError(f"A fixture has the same home and away team in {path}")
    if finished.duplicated(["_home_key", "_away_key"]).any():
        duplicated = finished.loc[finished.duplicated(["_home_key", "_away_key"], keep=False),
                                  ["date", "hometeam", "awayteam"]]
        raise ValueError(f"Duplicate completed fixture pairs in {path}:\n{duplicated.head(10)}")

    if require_full_season:
        clubs = set(finished["_home_key"]) | set(finished["_away_key"])
        if len(clubs) != 20:
            raise ValueError(f"Expected 20 distinct clubs in {path}; found {len(clubs)}")

    records = []
    for idx, row in finished.iterrows():
        label = f"{season} row {idx + 2} ({row['hometeam']} vs {row['awayteam']})"
        kickoff = local_date_time_to_utc(row["date"], row["time"], label)
        home_shots = clean_optional_int(row.get("hs"), "HS", label)
        away_shots = clean_optional_int(row.get("as"), "AS", label)
        home_corners = clean_optional_int(row.get("hc"), "HC", label)
        away_corners = clean_optional_int(row.get("ac"), "AC", label)
        home_yellow = clean_optional_int(row.get("hy"), "HY", label)
        away_yellow = clean_optional_int(row.get("ay"), "AY", label)
        home_red = clean_optional_int(row.get("hr"), "HR", label)
        away_red = clean_optional_int(row.get("ar"), "AR", label)
        referee_val = row.get("referee")
        referee = str(referee_val).strip() if referee_val is not None and pd.notna(referee_val) and str(referee_val).strip() else None
        records.append({
            "season": season,
            "date": kickoff.date().isoformat(),
            "home_name": str(row["hometeam"]).strip(),
            "away_name": str(row["awayteam"]).strip(),
            "kickoff_utc": kickoff,
            "home_goals": int(row["fthg"]),
            "away_goals": int(row["ftag"]),
            "home_shots": home_shots,
            "away_shots": away_shots,
            "home_corners": home_corners,
            "away_corners": away_corners,
            "home_cards": home_yellow + home_red if home_yellow is not None and home_red is not None else None,
            "away_cards": away_yellow + away_red if away_yellow is not None and away_red is not None else None,
            "referee": referee,
        })
    return records


def prepare_records(conn, team_map: dict[str, int], season_files: list[tuple[str, Path, bool]]) -> list[dict]:
    all_records: list[dict] = []
    unmapped: set[str] = set()
    for season, path, require_full_season in season_files:
        records = _read_season(path, season, require_full_season)
        logging.info("Read %s: %d completed matches", season, len(records))
        for record in records:
            home_id = resolve_team(record["home_name"], team_map)
            away_id = resolve_team(record["away_name"], team_map)
            if home_id is None:
                unmapped.add(record["home_name"])
            if away_id is None:
                unmapped.add(record["away_name"])
            if home_id is not None and away_id is not None and home_id == away_id:
                raise ValueError(f"Home and away resolved to same team ID: {record['home_name']} vs {record['away_name']}")
            record["home_team_id"] = home_id
            record["away_team_id"] = away_id
        all_records.extend(records)
    if unmapped:
        raise ValueError(
            "These source club names could not be mapped uniquely to `teams.id`: "
            f"{sorted(unmapped)}. No writes were made. Add verified aliases to the teams table "
            "or extend SOURCE_TEAM_ALIASES, then rerun the dry-run."
        )
    keys = [(r["season"], r["home_team_id"], r["away_team_id"]) for r in all_records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (season, home team, away team) keys across the input files; no writes made")
    return all_records


def full_season_label(value: object) -> str | None:
    """Convert FPL labels such as 2025-26 or 2025-2026 to 2025-2026."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})\s*[-/]\s*(\d{2}|\d{4})", text)
    if not match:
        return None
    start = int(match.group(1))
    end_raw = match.group(2)
    end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
    if end < start:
        end += 100
    return f"{start}-{end}"


def load_fpl_gameweek_map(path: Path, team_map: dict[str, int]) -> dict[tuple[str, int, int], int]:
    """Use existing FPL history to fill matchweek where the results CSV lacks it."""
    if not path.exists():
        logging.warning("Player history not found at %s; gameweek mapping will be unavailable", path)
        return {}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read player history {path}: {exc}") from exc
    required = {"season", "fixture_id", "player_team", "gameweek", "was_home"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Player history {path} is missing columns required for gameweek mapping: {sorted(missing)}")

    df["full_season"] = df["season"].map(full_season_label)
    df = df[df["full_season"].isin({"2025-2026", "2026-2027"})].copy()
    df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce")
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["_home"] = df["was_home"].astype(str).str.strip().str.casefold().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    )
    df = df[df["fixture_id"].notna() & df["gameweek"].notna() & df["_home"].notna()]
    mapping: dict[tuple[str, int, int], int] = {}

    for (season, fixture_id), group in df.groupby(["full_season", "fixture_id"], sort=False):
        gameweeks = sorted(set(int(v) for v in group["gameweek"].dropna() if float(v).is_integer()))
        if len(gameweeks) != 1:
            continue
        home_names = group.loc[group["_home"].eq(True), "player_team"].dropna().astype(str)
        away_names = group.loc[group["_home"].eq(False), "player_team"].dropna().astype(str)
        if home_names.empty or away_names.empty:
            continue
        home_name = home_names.value_counts().index[0]
        away_name = away_names.value_counts().index[0]
        home_id = resolve_team(home_name, team_map)
        away_id = resolve_team(away_name, team_map)
        if home_id is None or away_id is None or home_id == away_id:
            continue
        key = (season, home_id, away_id)
        previous = mapping.get(key)
        if previous is not None and previous != gameweeks[0]:
            # Conflicting gameweek assignments are treated as unknown.
            mapping.pop(key, None)
            continue
        mapping[key] = gameweeks[0]
    return mapping


def matchweek_is_nullable(conn) -> bool | None:
    """Return True/False for matches.matchweek nullability; None if not found."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_schema = ANY(current_schemas(false))
              AND table_name = 'matches' AND column_name = 'matchweek'
            ORDER BY array_position(current_schemas(false), table_schema)
            LIMIT 1
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    return str(row[0]).upper() == "YES"

def inspect_existing(conn, records: list[dict]) -> tuple[set[tuple[str, int, int]], set[int], dict[tuple[str, int, int], dict]]:
    seasons = sorted({r["season"] for r in records})
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, season, home_team_id, away_team_id, kickoff_utc, matchweek "
            "FROM matches WHERE season = ANY(%s)",
            (seasons,),
        )
        existing_rows = cur.fetchall()
        existing = {}
        for match_id, season, home_id, away_id, kickoff, matchweek in existing_rows:
            existing[(season, int(home_id), int(away_id))] = {
                "id": int(match_id), "kickoff_utc": kickoff, "matchweek": matchweek
            }
        matching_ids = [existing[(r["season"], r["home_team_id"], r["away_team_id"])]["id"]
                        for r in records if (r["season"], r["home_team_id"], r["away_team_id"]) in existing]
        prediction_match_ids = set()
        if matching_ids:
            cur.execute("SELECT DISTINCT match_id FROM predictions WHERE match_id = ANY(%s)", (matching_ids,))
            prediction_match_ids = {int(row[0]) for row in cur.fetchall()}
    return set(existing.keys()), prediction_match_ids, existing


UPSERT_SQL = """
INSERT INTO matches (
    season, matchweek, kickoff_utc, home_team_id, away_team_id, status,
    home_goals, away_goals, home_shots, away_shots,
    home_corners, away_corners, home_cards, away_cards, referee
)
VALUES (%s, %s, %s, %s, %s, 'FINISHED', %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (season, home_team_id, away_team_id)
DO UPDATE SET
    matchweek = COALESCE(matches.matchweek, EXCLUDED.matchweek),
    kickoff_utc = CASE
        WHEN EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = matches.id)
            THEN matches.kickoff_utc
        ELSE EXCLUDED.kickoff_utc
    END,
    status = 'FINISHED',
    home_goals = EXCLUDED.home_goals,
    away_goals = EXCLUDED.away_goals,
    home_shots = COALESCE(EXCLUDED.home_shots, matches.home_shots),
    away_shots = COALESCE(EXCLUDED.away_shots, matches.away_shots),
    home_corners = COALESCE(EXCLUDED.home_corners, matches.home_corners),
    away_corners = COALESCE(EXCLUDED.away_corners, matches.away_corners),
    home_cards = COALESCE(EXCLUDED.home_cards, matches.home_cards),
    away_cards = COALESCE(EXCLUDED.away_cards, matches.away_cards),
    referee = COALESCE(EXCLUDED.referee, matches.referee)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season-2526", type=Path, default=DEFAULT_2526,
                        help=f"2025-26 E0 CSV (default: {DEFAULT_2526})")
    parser.add_argument("--season-2627", type=Path, default=DEFAULT_2627,
                        help=f"2026-27 E0 CSV (default: {DEFAULT_2627})")
    parser.add_argument("--player-history", type=Path, default=ROOT / "fpl_complete_fixture_history.csv",
                        help="optional FPL history CSV used to recover matchweek/gameweek")
    parser.add_argument("--download", action="store_true", help="download both latest E0 CSV files into data/raw")
    parser.add_argument("--apply", action="store_true", help="commit changes; without this flag, run is dry-run")
    args = parser.parse_args()

    try:
        if args.download:
            download_csv("2526", args.season_2526)
            download_csv("2627", args.season_2627)

        season_files = [
            ("2025-2026", args.season_2526, True),
            ("2026-2027", args.season_2627, False),
        ]
        conn = get_db_connection()
        try:
            team_map = load_team_mappings(conn)
            records = prepare_records(conn, team_map, season_files)
            fpl_gameweeks = load_fpl_gameweek_map(args.player_history, team_map)
            fpl_week_filled = 0
            for rec in records:
                key = (rec["season"], rec["home_team_id"], rec["away_team_id"])
                rec["fpl_matchweek"] = fpl_gameweeks.get(key)
                if rec["fpl_matchweek"] is not None:
                    fpl_week_filled += 1
            existing_keys, predicted_match_ids, existing = inspect_existing(conn, records)

            n_existing = sum((r["season"], r["home_team_id"], r["away_team_id"]) in existing_keys for r in records)
            n_new = len(records) - n_existing
            n_existing_with_predictions = sum(
                existing[(r["season"], r["home_team_id"], r["away_team_id"])]["id"] in predicted_match_ids
                for r in records
                if (r["season"], r["home_team_id"], r["away_team_id"]) in existing
            )
            matchweek_nullable = matchweek_is_nullable(conn)
            missing_required_matchweeks = []
            for rec in records:
                key = (rec["season"], rec["home_team_id"], rec["away_team_id"])
                prior = existing.get(key)
                stored_week = prior["matchweek"] if prior else None
                if stored_week is None and rec["fpl_matchweek"] is None:
                    missing_required_matchweeks.append(key)
            if matchweek_nullable is None:
                raise ValueError("Could not verify matches.matchweek schema; no writes made")
            if not matchweek_nullable and missing_required_matchweeks:
                raise ValueError(
                    f"matches.matchweek is NOT NULL, but {len(missing_required_matchweeks)} rows have no existing or "
                    "FPL-derived matchweek. Ensure fpl_complete_fixture_history.csv includes these fixtures; no writes made."
                )

            print("\nCOMPLETED-RESULTS SYNC PRECHECK")
            print("Mode:", "APPLY (database changes will be committed)" if args.apply else "DRY-RUN (no database writes)")
            print("Source files:")
            print(f"  2025-2026: {args.season_2526} ({sum(r['season'] == '2025-2026' for r in records)} completed rows; expected 380)")
            print(f"  2026-2027: {args.season_2627} ({sum(r['season'] == '2026-2027' for r in records)} completed rows; unplayed rows ignored)")
            print(f"Total completed-result rows to upsert: {len(records)}")
            print(f"Existing fixtures to update: {n_existing}")
            print(f"New historical fixtures to insert: {n_new}")
            print(f"Existing fixture rows with prediction records: {n_existing_with_predictions}")
            print(f"FPL-derived matchweeks available: {fpl_week_filled}")
            print(f"matches.matchweek nullable: {matchweek_nullable}")
            for season, _, _ in season_files:
                season_records = [r for r in records if r["season"] == season]
                if season_records:
                    dates = sorted(r["date"] for r in season_records)
                    print(f"  {season} source date span: {dates[0]} to {dates[-1]}")
            print("Predictions table: no prediction rows will be inserted, updated, or deleted.")

            if not args.apply:
                conn.rollback()
                print("\nDry-run complete. Review counts and dates; back up the database before rerunning with --apply.")
                return 0

            # One transaction for both seasons: either every validated result is applied or none is.
            with conn.cursor() as cur:
                for rec in records:
                    key = (rec["season"], rec["home_team_id"], rec["away_team_id"])
                    prior = existing.get(key)
                    # Preserve an existing matchweek; otherwise use the FPL fixture's gameweek if available.
                    matchweek = prior["matchweek"] if prior and prior["matchweek"] is not None else rec["fpl_matchweek"]
                    cur.execute(UPSERT_SQL, (
                        rec["season"], matchweek, rec["kickoff_utc"], rec["home_team_id"], rec["away_team_id"],
                        rec["home_goals"], rec["away_goals"], rec["home_shots"], rec["away_shots"],
                        rec["home_corners"], rec["away_corners"], rec["home_cards"], rec["away_cards"], rec["referee"],
                    ))
            conn.commit()
            print("\nSync committed successfully.")
            print("Next: run `python -m src.score_predictions` to score saved predictions for matches now marked FINISHED.")
            return 0
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    except Exception as exc:
        logging.error("Sync aborted before commit: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
