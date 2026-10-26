#!/usr/bin/env python3
"""Audit historical FPL player-fixture data before player-availability modelling.

Run from the repository root:
    python -m src.audit_player_history
    python -m src.audit_player_history --input data/fpl_complete_fixture_history.csv

This script is read-only. It creates CSV audit reports and does not alter the database.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("fpl_complete_fixture_history.csv")
DEFAULT_OUT_DIR = Path("reports") / "player_availability_audit"
DEFAULT_SEASONS = ("2022-23", "2023-24", "2024-25")
REQUIRED_COLUMNS = {
    "season", "player_code", "element_id", "player_name", "position",
    "player_team", "opponent_team", "fixture_id", "kickoff_utc", "was_home",
    "minutes", "started", "goals", "assists", "xG", "xA", "team_source",
}


def norm_season(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{2}|\d{4})", raw)
    if not m:
        return raw
    start, end = m.groups()
    return f"{start}-{end[-2:]}"


def parse_bool(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False,
               "1.0": True, "0.0": False, "yes": True, "no": False}
    return s.map(mapping).astype("boolean")


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to fpl_complete_fixture_history.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for generated audit reports")
    parser.add_argument("--seasons", nargs="*", default=list(DEFAULT_SEASONS),
                        help="Season labels, e.g. 2022-23 2023-24 2024-25")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: player history CSV not found: {args.input}", file=sys.stderr)
        print("Pass its actual path with --input.", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        print("ERROR: input is missing required columns: " + ", ".join(missing), file=sys.stderr)
        return 2

    df["season_norm"] = df["season"].map(norm_season)
    wanted = {norm_season(s) for s in args.seasons}
    all_seasons = sorted(s for s in df["season_norm"].dropna().unique())
    selected = df[df["season_norm"].isin(wanted)].copy()
    if selected.empty:
        print(f"ERROR: no rows found for requested seasons {sorted(wanted)}", file=sys.stderr)
        print(f"Seasons present: {', '.join(all_seasons)}", file=sys.stderr)
        return 2

    selected["fixture_id"] = pd.to_numeric(selected["fixture_id"], errors="coerce")
    selected["minutes"] = pd.to_numeric(selected["minutes"], errors="coerce")
    selected["xG"] = pd.to_numeric(selected["xG"], errors="coerce")
    selected["xA"] = pd.to_numeric(selected["xA"], errors="coerce")
    selected["goals"] = pd.to_numeric(selected["goals"], errors="coerce")
    selected["assists"] = pd.to_numeric(selected["assists"], errors="coerce")
    selected["kickoff_utc"] = pd.to_datetime(selected["kickoff_utc"], utc=True, errors="coerce")
    selected["started_bool"] = parse_bool(selected["started"])
    selected["was_home_bool"] = parse_bool(selected["was_home"])
    selected["player_team"] = selected["player_team"].astype("string").str.strip()
    selected["player_name"] = selected["player_name"].astype("string").str.strip()

    # A season-scoped fallback avoids falsely merging different element IDs across seasons.
    code = selected["player_code"].astype("string").str.strip()
    elem = selected["element_id"].astype("string").str.strip()
    name = selected["player_name"].str.lower().str.replace(r"\s+", " ", regex=True)
    selected["player_key"] = "name:" + name
    has_code = code.notna() & code.ne("") & code.ne("<NA>")
    selected.loc[has_code, "player_key"] = "code:" + code[has_code]
    no_code = ~has_code
    has_elem = no_code & elem.notna() & elem.ne("") & elem.ne("<NA>")
    selected.loc[has_elem, "player_key"] = (
        "element:" + selected.loc[has_elem, "season_norm"].astype(str) + ":" + elem[has_elem]
    )

    # Duplicates using the best available player identity for the season/fixture.
    dup_mask = selected.duplicated(["season_norm", "fixture_id", "player_key"], keep=False)
    duplicate_rows = selected.loc[dup_mask].copy()

    # One record per team in a fixture. We only consider lineup coverage valid if
    # start flags are known for all observed player rows and exactly 11 are True.
    group_keys = ["season_norm", "fixture_id", "kickoff_utc", "player_team"]
    team_rows = []
    for key, g in selected.groupby(group_keys, dropna=False, sort=False):
        season, fixture_id, kickoff, team = key
        known = int(g["started_bool"].notna().sum())
        starter_count = int(g["started_bool"].fillna(False).sum())
        missing_started = int(g["started_bool"].isna().sum())
        rows = len(g)
        valid = (pd.notna(fixture_id) and pd.notna(kickoff) and pd.notna(team)
                 and known == rows and missing_started == 0 and starter_count == 11)
        starters = g[g["started_bool"].eq(True)]
        starter_xgi_known = int((starters["xG"].notna() & starters["xA"].notna()).sum())
        team_rows.append({
            "season": season, "fixture_id": fixture_id, "kickoff_utc": kickoff,
            "team": team, "player_rows": rows, "started_known_rows": known,
            "started_missing_rows": missing_started, "starter_count": starter_count,
            "valid_exactly_11_starters": bool(valid),
            "starter_xG_xA_both_known": starter_xgi_known,
            "starter_xG_xA_coverage": safe_rate(starter_xgi_known, starter_count),
            "team_source_fixture_share": float((g["team_source"] == "fixture").mean()),
        })
    team_report = pd.DataFrame(team_rows)

    # Fixture coverage is counted only when two distinct teams are present and both
    # have valid XI records. This is a readiness audit, not a claim about injuries.
    fixture_rows = []
    for (season, fixture_id), g in team_report.groupby(["season", "fixture_id"], dropna=False, sort=False):
        teams = g["team"].dropna().astype(str).unique()
        valid_teams = int(g["valid_exactly_11_starters"].sum())
        home_count = int((selected["season_norm"].eq(season) & selected["fixture_id"].eq(fixture_id) & selected["was_home_bool"].eq(True)).sum())
        away_count = int((selected["season_norm"].eq(season) & selected["fixture_id"].eq(fixture_id) & selected["was_home_bool"].eq(False)).sum())
        # A fixture needs exactly two distinct teams and two valid team XIs.
        ready = len(teams) == 2 and len(g) == 2 and valid_teams == 2 and home_count > 0 and away_count > 0
        kickoff_vals = g["kickoff_utc"].dropna()
        fixture_rows.append({
            "season": season, "fixture_id": fixture_id,
            "kickoff_utc": kickoff_vals.iloc[0] if not kickoff_vals.empty else pd.NaT,
            "teams_observed": " | ".join(sorted(teams)), "team_groups": len(g),
            "teams_with_valid_XI": valid_teams, "home_side_rows": home_count,
            "away_side_rows": away_count, "lineup_ready_both_teams": bool(ready),
        })
    fixture_report = pd.DataFrame(fixture_rows)

    summaries = []
    for season, g in selected.groupby("season_norm", sort=True):
        fg = fixture_report[fixture_report["season"].eq(season)]
        tg = team_report[team_report["season"].eq(season)]
        season_dup = duplicate_rows[duplicate_rows["season_norm"].eq(season)]
        starts_known = int(g["started_bool"].notna().sum())
        xg_known = int(g["xG"].notna().sum())
        xa_known = int(g["xA"].notna().sum())
        xgi_both = int((g["xG"].notna() & g["xA"].notna()).sum())
        player_fixture_count = int(g.dropna(subset=["fixture_id", "player_key"]).groupby(["fixture_id", "player_key"]).ngroups)
        summaries.append({
            "season": season,
            "rows": len(g),
            "fixtures_in_file": int(g["fixture_id"].nunique()),
            "kickoff_missing_rate": float(g["kickoff_utc"].isna().mean()),
            "started_known_rate": safe_rate(starts_known, len(g)),
            "xG_known_rate": safe_rate(xg_known, len(g)),
            "xA_known_rate": safe_rate(xa_known, len(g)),
            "xG_xA_both_known_rate": safe_rate(xgi_both, len(g)),
            "team_name_missing_rate": float(g["player_team"].isna().mean()),
            "duplicate_player_fixture_rows": len(season_dup),
            "team_fixture_groups": len(tg),
            "team_groups_with_exactly_11_known_starters": int(tg["valid_exactly_11_starters"].sum()),
            "team_XI_valid_rate": float(tg["valid_exactly_11_starters"].mean()) if len(tg) else None,
            "fixtures_with_both_XIs_ready": int(fg["lineup_ready_both_teams"].sum()),
            "fixture_XI_ready_rate": float(fg["lineup_ready_both_teams"].mean()) if len(fg) else None,
        })
    summary = pd.DataFrame(summaries)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "player_history_season_summary.csv"
    team_path = args.out_dir / "player_history_team_lineups.csv"
    fixture_path = args.out_dir / "player_history_fixture_readiness.csv"
    dup_path = args.out_dir / "player_history_duplicate_rows.csv"
    summary.to_csv(summary_path, index=False)
    team_report.to_csv(team_path, index=False)
    fixture_report.to_csv(fixture_path, index=False)
    duplicate_rows.to_csv(dup_path, index=False)

    print("PLAYER HISTORY READINESS AUDIT (read-only)")
    print(f"Input: {args.input}")
    print(f"Seasons requested: {', '.join(sorted(wanted))}")
    print(f"Rows inspected: {len(selected):,}")
    print("\nSeason summary (rates range from 0 to 1):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary.round(4).to_string(index=False))
    print("\nKey interpretation:")
    print("  - `fixture_XI_ready_rate` is the share of fixture IDs with exactly 11 known starters")
    print("    for each of two observed teams, known home/away roles, and kickoff timestamps.")
    print("  - `xG_xA_both_known_rate` measures source-field coverage, not whether values are accurate.")
    print("  - This audit does not infer injuries: a player not starting may be benched, rotated, suspended, or absent.")
    print("\nReports saved:")
    print(f"  Summary:  {summary_path}")
    print(f"  Team XIs: {team_path}")
    print(f"  Fixtures: {fixture_path}")
    print(f"  Duplicates: {dup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
