#!/usr/bin/env python3
"""
Build one clean table of player-fixture rows from a local copy of the
community-maintained Fantasy-Premier-League archive.

Check the archive's licence and credit it in your README.

Output columns
    season, player_code, element_id, player_name, position,
    player_team_id, player_team, opponent_team_id, opponent_team,
    fixture_id, gameweek, kickoff_utc, was_home,
    minutes, started, goals, assists, xG, xA, team_source

Notes
  * `started` is missing (blank) where the source has no `starts` column. It is
    never turned into False.
  * xG / xA are missing (blank) where the source has no expected_* columns.
  * The player's team for each row comes from the fixture (team_h / team_a),
    so mid-season transfers are handled. `team_source` says where it came from:
    "fixture", "derived_from_opponents" (rebuilt from the players' own
    opponent_team + was_home, used when fixtures.csv is missing),
    "season_default" (players_raw team, may be wrong for transfers) or "unknown".
  * FPL team ids are per season. Do not compare ids across seasons; use the
    names in team_names_seen.csv to build a mapping to your own `teams` table.

Usage
    python build_fpl_player_history.py --show-columns        # inspect files first
    python build_fpl_player_history.py                       # run with defaults
    python build_fpl_player_history.py --data-path "D:\\fpl\\data" --out-dir out
    python build_fpl_player_history.py --seasons 2023-24 2024-25
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_BASE = (
    r"C:\Users\user\Downloads\Fantasy-Premier-League-master"
    r"\Fantasy-Premier-League-master\data"
)

SEASON_RE = re.compile(r"^(\d{4})\s*-\s*(\d{2})$")
POSITION_BY_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_BY_TEXT = {
    "GK": "GKP", "GKP": "GKP", "GKS": "GKP",
    "DEF": "DEF", "MID": "MID", "FWD": "FWD", "FW": "FWD",
}
TEAM_MINUTES_EXPECTED = 990  # 11 players x 90 minutes


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def read_csv_safe(path):
    """Read a CSV as UTF-8 first (so accents stay correct), then latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def normalise_season(name):
    """'2016- 17' / '2016-17' -> '2016-17'; anything else -> None."""
    match = SEASON_RE.match(name.strip())
    return f"{match.group(1)}-{match.group(2)}" if match else None


def discover_seasons(base_path):
    found = {}
    for entry in sorted(os.listdir(base_path)):
        full = os.path.join(base_path, entry)
        label = normalise_season(entry)
        if label and os.path.isdir(full):
            found[label] = full
    return found


def normalise_position(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return POSITION_BY_TYPE.get(int(value))
    except (TypeError, ValueError):
        return POSITION_BY_TEXT.get(str(value).strip().upper())


def clean_text(value):
    return str(value).strip() if pd.notna(value) else ""


def parse_folder(folder):
    """'Mohamed_Salah_253' -> (253, 'Mohamed Salah', 'mohamed_salah')."""
    tokens = folder.split("_")
    element_id = None
    if tokens and tokens[-1].isdigit():
        element_id = int(tokens[-1])
        tokens = tokens[:-1]
    return element_id, " ".join(tokens), "_".join(tokens).lower()


def name_key(first, second):
    return f"{clean_text(first)} {clean_text(second)}".strip().replace(" ", "_").lower()


def parse_bool(series):
    text = series.astype(str).str.strip().str.lower()
    return text.map({
        "true": True, "false": False,
        "1": True, "0": False, "1.0": True, "0.0": False,
    })


# ----------------------------------------------------------------------------
# Per-season lookup tables (teams, fixtures, player metadata)
# ----------------------------------------------------------------------------
class SeasonContext:
    def __init__(self, season, season_dir):
        self.season = season
        self.dir = season_dir
        self.team_names = {}
        self.fixtures = None
        self.raw_by_id = {}
        self.raw_by_name = {}
        self.clean_by_name = {}
        self.notes = []
        self._load_teams()
        self._load_fixtures()
        self._load_players_raw()
        self._load_cleaned_players()

    def _path(self, name):
        return os.path.join(self.dir, name)

    def _load_teams(self):
        path = self._path("teams.csv")
        if not os.path.exists(path):
            self.notes.append("teams.csv missing: team names will be Team_<id>")
            return
        try:
            df = read_csv_safe(path)
            self.team_names = {
                int(i): str(n) for i, n in zip(df["id"], df["name"]) if pd.notna(i)
            }
        except Exception as exc:
            self.notes.append(f"teams.csv could not be read: {exc!r}")

    def _load_fixtures(self):
        path = self._path("fixtures.csv")
        if not os.path.exists(path):
            self.notes.append("fixtures.csv missing: teams fall back to season default")
            return
        try:
            df = read_csv_safe(path)
            kickoff = (
                pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
                if "kickoff_time" in df.columns
                else pd.Series(pd.NaT, index=df.index)
            )
            fx = pd.DataFrame({
                "team_h": pd.to_numeric(df["team_h"], errors="coerce"),
                "team_a": pd.to_numeric(df["team_a"], errors="coerce"),
                "kickoff": kickoff,
            })
            fx.index = pd.to_numeric(df["id"], errors="coerce")
            fx = fx[fx.index.notna()]
            fx.index = fx.index.astype(int)
            self.fixtures = fx[~fx.index.duplicated(keep="first")]
        except Exception as exc:
            self.notes.append(f"fixtures.csv could not be used: {exc!r}")

    def _load_players_raw(self):
        path = self._path("players_raw.csv")
        if not os.path.exists(path):
            self.notes.append("players_raw.csv missing: no player_code / position from it")
            return
        try:
            df = read_csv_safe(path)
            for _, row in df.iterrows():
                code = row.get("code")
                meta = {
                    "code": int(code) if pd.notna(code) else None,
                    "position": normalise_position(row.get("element_type")),
                    "team": row.get("team"),
                }
                if pd.notna(row.get("id")):
                    self.raw_by_id[int(row["id"])] = meta
                key = name_key(row.get("first_name"), row.get("second_name"))
                if key:
                    self.raw_by_name.setdefault(key, meta)
        except Exception as exc:
            self.notes.append(f"players_raw.csv could not be used: {exc!r}")

    def _load_cleaned_players(self):
        path = self._path("cleaned_players.csv")
        if not os.path.exists(path):
            return
        try:
            df = read_csv_safe(path)
            if "element_type" not in df.columns:
                return
            for _, row in df.iterrows():
                key = name_key(row.get("first_name"), row.get("second_name"))
                position = normalise_position(row.get("element_type"))
                if key and position:
                    self.clean_by_name.setdefault(
                        key, {"code": None, "position": position, "team": np.nan}
                    )
        except Exception as exc:
            self.notes.append(f"cleaned_players.csv could not be used: {exc!r}")


# ----------------------------------------------------------------------------
# One player's gw.csv -> tidy rows
# ----------------------------------------------------------------------------
def process_player(season, folder, gw_path, ctx):
    gw = read_csv_safe(gw_path)
    if gw.empty:
        return None, 0
    if "minutes" not in gw.columns:
        raise ValueError("no 'minutes' column")

    element_id, player_name, key = parse_folder(folder)
    meta = (
        ctx.raw_by_id.get(element_id)
        or ctx.raw_by_name.get(key)
        or ctx.clean_by_name.get(key)
        or {}
    )

    idx = gw.index

    def col(name):
        return gw[name] if name in gw.columns else pd.Series(np.nan, index=idx)

    fixture = pd.to_numeric(col("fixture"), errors="coerce")
    opponent_raw = pd.to_numeric(col("opponent_team"), errors="coerce")
    minutes = pd.to_numeric(col("minutes"), errors="coerce")

    # Fixture lookups (own / opponent team ids and kickoff time)
    if ctx.fixtures is not None:
        fx_home = fixture.map(ctx.fixtures["team_h"])
        fx_away = fixture.map(ctx.fixtures["team_a"])
        fx_kickoff = fixture.map(ctx.fixtures["kickoff"])
    else:
        fx_home = pd.Series(np.nan, index=idx)
        fx_away = pd.Series(np.nan, index=idx)
        fx_kickoff = pd.Series(pd.NaT, index=idx)

    # Home / away: use the source flag, else derive it from the fixture
    was_home = parse_bool(col("was_home"))
    derived = pd.Series(
        np.where(opponent_raw.notna() & fx_away.notna(), opponent_raw == fx_away, np.nan),
        index=idx,
    ).map({1.0: True, 0.0: False})
    was_home = was_home.where(was_home.notna(), derived)
    home_mask = was_home.eq(True)
    away_mask = was_home.eq(False)

    own_id = pd.Series(np.nan, index=idx)
    opp_id = pd.Series(np.nan, index=idx)
    own_id = own_id.mask(home_mask, fx_home).mask(away_mask, fx_away)
    opp_id = opp_id.mask(home_mask, fx_away).mask(away_mask, fx_home)
    team_source = pd.Series("unknown", index=idx)
    team_source = team_source.mask(own_id.notna(), "fixture")

    default_team = pd.to_numeric(pd.Series([meta.get("team", np.nan)]), errors="coerce").iloc[0]
    need_team = own_id.isna()
    if pd.notna(default_team):
        own_id = own_id.mask(need_team, default_team)
        team_source = team_source.mask(need_team, "season_default")
    opp_id = opp_id.fillna(opponent_raw)

    kickoff = pd.to_datetime(fx_kickoff, utc=True, errors="coerce")
    kickoff = kickoff.fillna(pd.to_datetime(col("kickoff_time"), utc=True, errors="coerce"))

    # started stays missing for seasons without a `starts` column
    starts = pd.to_numeric(col("starts"), errors="coerce")
    started = (starts == 1).astype("boolean").mask(starts.isna())

    out = pd.DataFrame({
        "season": season,
        "player_code": meta.get("code"),
        "element_id": element_id,
        "player_name": player_name,
        "position": meta.get("position"),
        "player_team_id": pd.to_numeric(own_id, errors="coerce").astype("Int64"),
        "opponent_team_id": pd.to_numeric(opp_id, errors="coerce").astype("Int64"),
        "fixture_id": fixture.astype("Int64"),
        "gameweek": pd.to_numeric(col("round"), errors="coerce").astype("Int64"),
        "kickoff_utc": kickoff,
        "was_home": was_home,
        "minutes": minutes,
        "started": started,
        "goals": pd.to_numeric(col("goals_scored"), errors="coerce"),
        "assists": pd.to_numeric(col("assists"), errors="coerce"),
        "xG": pd.to_numeric(col("expected_goals"), errors="coerce"),
        "xA": pd.to_numeric(col("expected_assists"), errors="coerce"),
        "team_source": team_source,
    })

    usable = out["fixture_id"].notna() & out["minutes"].notna()
    dropped = int((~usable).sum())
    return out[usable], dropped


# ----------------------------------------------------------------------------
# Post-processing and diagnostics
# ----------------------------------------------------------------------------
def derive_missing_fixture_teams(df):
    """
    For rows whose team did not come from fixtures.csv, rebuild each fixture's
    home / away team ids from the players' own `opponent_team` + `was_home`:
    away players' opponent is the home team, home players' opponent is the away
    team. Needs no external file. Returns (df, number_of_rows_fixed).
    """
    need = df["team_source"] != "fixture"
    if not need.any():
        return df, 0

    work = df[df["was_home"].notna() & df["opponent_team_id"].notna() & df["fixture_id"].notna()]
    group_keys = ["season", "fixture_id"]

    def mode_map(rows):
        grouped = rows.groupby(group_keys)["opponent_team_id"].agg(lambda s: int(s.mode().iloc[0]))
        return {(season, int(fid)): team for (season, fid), team in grouped.items()}

    # away players' opponent = home team; home players' opponent = away team
    home_map = mode_map(work[work["was_home"].eq(False)])
    away_map = mode_map(work[work["was_home"].eq(True)])

    idx = df.index[need]
    sub = df.loc[idx, ["season", "fixture_id", "was_home"]]
    pairs = list(zip(sub["season"], sub["fixture_id"]))
    home_ids = pd.Series([home_map.get((s, int(f))) if pd.notna(f) else None for s, f in pairs],
                         index=idx, dtype="float64")
    away_ids = pd.Series([away_map.get((s, int(f))) if pd.notna(f) else None for s, f in pairs],
                         index=idx, dtype="float64")

    is_home = sub["was_home"].eq(True)
    is_away = sub["was_home"].eq(False)
    own = pd.Series(np.nan, index=idx).mask(is_home, home_ids).mask(is_away, away_ids)
    opp = pd.Series(np.nan, index=idx).mask(is_home, away_ids).mask(is_away, home_ids)

    ok = own.notna() & opp.notna()
    fixed_idx = idx[ok.to_numpy()]
    df.loc[fixed_idx, "player_team_id"] = own[ok].astype("Int64")
    df.loc[fixed_idx, "opponent_team_id"] = opp[ok].astype("Int64")
    df.loc[fixed_idx, "team_source"] = "derived_from_opponents"
    return df, int(ok.sum())


def attach_team_names(df, names_by_season):
    for id_col, name_col in (
        ("player_team_id", "player_team"),
        ("opponent_team_id", "opponent_team"),
    ):
        df[name_col] = None
        for season, mapping in names_by_season.items():
            mask = (df["season"] == season) & df[id_col].notna()
            if mask.any():
                df.loc[mask, name_col] = df.loc[mask, id_col].map(
                    lambda i, m=mapping: m.get(int(i), f"Team_{int(i)}")
                )
    return df


def run_diagnostics(df):
    d = df.copy()
    d["team_unknown"] = d["player_team"].isna() | d["player_team"].astype(str).str.startswith("Team_")

    summary = d.groupby("season").agg(
        rows=("minutes", "size"),
        fixtures=("fixture_id", "nunique"),
        code_missing=("player_code", lambda s: s.isna().mean()),
        position_missing=("position", lambda s: s.isna().mean()),
        team_unknown=("team_unknown", "mean"),
        team_from_fixture=("team_source", lambda s: (s == "fixture").mean()),
        team_derived=("team_source", lambda s: (s == "derived_from_opponents").mean()),
        kickoff_missing=("kickoff_utc", lambda s: s.isna().mean()),
        started_known=("started", lambda s: s.notna().mean()),
        xg_known=("xG", lambda s: s.notna().mean()),
    )
    print("\n--- Per-season summary (rates are 0..1; expect 380 fixtures in a full season) ---")
    print(summary.round(3).to_string())

    keyed = d.dropna(subset=["player_team_id"])
    team_match = keyed.groupby(["season", "fixture_id", "player_team_id"])

    minutes_total = team_match["minutes"].sum()
    off = ((minutes_total - TEAM_MINUTES_EXPECTED).abs() > 60).groupby("season").mean()
    print(f"\n--- Share of team-matches whose minutes total is >60 away from {TEAM_MINUTES_EXPECTED} ---")
    print("(low is good; the in-progress season and red-card games add a little)")
    print(off.round(3).to_string())

    proxy = keyed.assign(played60=keyed["minutes"] >= 60).groupby(
        ["season", "fixture_id", "player_team_id"]
    )["played60"].sum()
    proxy_ok = proxy.between(9, 11).groupby("season").mean()
    print("\n--- Share of team-matches with 9 to 11 players at 60+ minutes (starter proxy) ---")
    print(proxy_ok.round(3).to_string())

    known = keyed[keyed["started"].notna()]
    if not known.empty:
        starters = known.groupby(["season", "fixture_id", "player_team_id"])["started"].agg(
            lambda s: int(s.sum())
        )
        exactly_11 = (starters == 11).groupby("season").mean()
        print("\n--- Share of team-matches with exactly 11 flagged starters (seasons with `starts`) ---")
        print(exactly_11.round(3).to_string())


def show_columns(seasons):
    for season, season_dir in seasons.items():
        print(f"\n=== {season} ({season_dir}) ===")
        print("top-level files:", sorted(
            f for f in os.listdir(season_dir) if os.path.isfile(os.path.join(season_dir, f))
        ))
        for name in ("teams.csv", "fixtures.csv", "players_raw.csv", "cleaned_players.csv"):
            path = os.path.join(season_dir, name)
            if os.path.exists(path):
                print(f"{name}: {list(read_csv_safe(path).columns)}")
        players_dir = os.path.join(season_dir, "players")
        if os.path.isdir(players_dir):
            for folder in sorted(os.listdir(players_dir)):
                gw_path = os.path.join(players_dir, folder, "gw.csv")
                if os.path.isfile(gw_path):
                    print(f"players/{folder}/gw.csv: {list(read_csv_safe(gw_path).columns)}")
                    break


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default=DEFAULT_BASE, help="folder that contains the season folders")
    parser.add_argument("--out-dir", default=".", help="where to write the CSV outputs")
    parser.add_argument("--seasons", nargs="*", help="optional season labels, e.g. 2023-24 2024-25")
    parser.add_argument("--show-columns", action="store_true", help="print file columns per season and exit")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        sys.exit(f"Data path not found: {args.data_path}")

    seasons = discover_seasons(args.data_path)
    if args.seasons:
        wanted = {normalise_season(s) or s for s in args.seasons}
        seasons = {k: v for k, v in seasons.items() if k in wanted}
    if not seasons:
        sys.exit("No season folders found (expected names like '2023-24').")
    print(f"Seasons found: {', '.join(seasons)}")

    if args.show_columns:
        show_columns(seasons)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    frames, failures, teams_seen = [], [], []
    names_by_season = {}
    dropped_total = 0

    for season, season_dir in seasons.items():
        players_dir = os.path.join(season_dir, "players")
        if not os.path.isdir(players_dir):
            print(f"Skipping {season}: no 'players' folder")
            continue

        ctx = SeasonContext(season, season_dir)
        names_by_season[season] = ctx.team_names
        for team_id, team_name in ctx.team_names.items():
            teams_seen.append({"season": season, "fpl_team_id": team_id, "fpl_team_name": team_name})
        for note in ctx.notes:
            print(f"  [{season}] note: {note}")

        season_rows = 0
        for folder in sorted(os.listdir(players_dir)):
            gw_path = os.path.join(players_dir, folder, "gw.csv")
            if not os.path.isfile(gw_path):
                continue
            try:
                rows, dropped = process_player(season, folder, gw_path, ctx)
            except Exception as exc:
                failures.append({"season": season, "folder": folder, "error": repr(exc)})
                continue
            dropped_total += dropped
            if rows is not None and not rows.empty:
                frames.append(rows)
                season_rows += len(rows)
        print(f"Parsed {season}: {season_rows:,} rows")

    if not frames:
        sys.exit("No rows parsed. Run with --show-columns and check the folder layout.")

    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["season", "element_id", "player_name", "fixture_id"])
    duplicates = before - len(df)

    df["player_code"] = pd.to_numeric(df["player_code"], errors="coerce").astype("Int64")
    df["element_id"] = pd.to_numeric(df["element_id"], errors="coerce").astype("Int64")
    df, derived_rows = derive_missing_fixture_teams(df)
    if derived_rows:
        print(f"Rebuilt team ids for {derived_rows:,} rows from opponent_team + was_home "
              f"(seasons without a usable fixtures.csv).")
    df = attach_team_names(df, names_by_season)

    ordered = [
        "season", "player_code", "element_id", "player_name", "position",
        "player_team_id", "player_team", "opponent_team_id", "opponent_team",
        "fixture_id", "gameweek", "kickoff_utc", "was_home",
        "minutes", "started", "goals", "assists", "xG", "xA", "team_source",
    ]
    df = df[ordered].sort_values(["season", "kickoff_utc", "fixture_id", "player_team_id"])

    main_path = os.path.join(args.out_dir, "fpl_complete_fixture_history.csv")
    teams_path = os.path.join(args.out_dir, "team_names_seen.csv")
    fail_path = os.path.join(args.out_dir, "fpl_failures.csv")
    df.to_csv(main_path, index=False)
    pd.DataFrame(teams_seen).to_csv(teams_path, index=False)
    pd.DataFrame(failures, columns=["season", "folder", "error"]).to_csv(fail_path, index=False)

    print(f"\nBuilt {len(df):,} player-fixture rows -> {main_path}")
    print(f"Dropped {dropped_total:,} rows with no fixture id or no minutes; "
          f"removed {duplicates:,} duplicate rows.")
    print(f"{len(failures)} player files failed (see {fail_path})")
    if failures:
        print("First failures:", failures[:5])
    print(f"Team names per season -> {teams_path} (use it to map to your own teams table)")

    run_diagnostics(df)


if __name__ == "__main__":
    main()