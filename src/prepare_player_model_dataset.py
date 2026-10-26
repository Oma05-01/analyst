#!/usr/bin/env python3
"""Join leakage-aware FPL lineup features to completed DB match outcomes (2022-23 onward).

Read-only with respect to PostgreSQL. This is a data-linkage/quality-check step;
it does not fit a model, create predictions, or modify locked prediction rows.

Run from the repository root:
    python -m src.prepare_player_model_dataset
    python -m src.prepare_player_model_dataset --features path/to/player_lineup_features.csv

Default outputs:
    data/processed/player_model_dataset.csv
    reports/player_model_data_join/unmatched_feature_rows.csv
    reports/player_model_data_join/unmatched_database_matches.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine

LOG = logging.getLogger("prepare_player_model_dataset")

MATCHES_SQL = text("""
    SELECT m.id AS match_id,
           m.season,
           m.matchweek,
           m.kickoff_utc AS database_kickoff_utc,
           m.status AS database_status,
           ht.name AS database_home_team,
           awt.name AS database_away_team,
           m.home_goals,
           m.away_goals
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams awt ON awt.id = m.away_team_id
    WHERE m.season IN ('2022-2023', '2023-2024', '2024-2025',
                       '2025-2026', '2026-2027',
                       '2022-23', '2023-24', '2024-25', '2025-26', '2026-27')
      AND m.status = 'FINISHED'
      AND m.home_goals IS NOT NULL
      AND m.away_goals IS NOT NULL
    ORDER BY m.season, m.kickoff_utc, m.id
""")

REQUIRED_FEATURE_COLUMNS = {
    "season", "fixture_id", "kickoff_utc", "home_team", "away_team",
    "home_xi_attack_xgi_per90_sum", "away_xi_attack_xgi_per90_sum",
    "home_xi_attack_delta_vs_recent_team_mean",
    "away_xi_attack_delta_vs_recent_team_mean",
    "lineup_strength_features_ready", "lineup_delta_features_ready",
}

# Canonicalise common naming differences between FPL archive and football-data imports.
ALIASES = {
    "afc bournemouth": "bournemouth",
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "ipswich town": "ipswich",
    "leicester city": "leicester",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "manchester utd": "manchester united",
    "newcastle united": "newcastle",
    "nott m forest": "nottingham forest",
    "nottm forest": "nottingham forest",
    "nott s forest": "nottingham forest",
    "notts forest": "nottingham forest",
    "sheffield utd": "sheffield united",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
    "sunderland afc": "sunderland",
    "hull city": "hull",
    "leeds united": "leeds",
}


def normalise_season(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    match = re.fullmatch(r"(\d{4})\s*[-/]\s*(\d{2}|\d{4})", s)
    if not match:
        return None
    start = int(match.group(1))
    tail = match.group(2)
    end = int(tail) if len(tail) == 4 else (start // 100) * 100 + int(tail)
    if end != start + 1:
        return None
    return f"{start}-{end}"


def base_team_key(value) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def team_key(value) -> str:
    key = base_team_key(value)
    return ALIASES.get(key, key)


def parse_flag(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().casefold() in {"true", "1", "yes"}
                      if pd.notna(value) else False)


def outcome_from_goals(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return "HOME"
    if home_goals < away_goals:
        return "AWAY"
    return "DRAW"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/player_lineup_features.csv",
                        help="output from build_player_lineup_features.py")
    parser.add_argument("--output", default="data/processed/player_model_dataset.csv",
                        help="joined dataset output CSV")
    parser.add_argument("--report-dir", default="reports/player_model_data_join",
                        help="directory for join audit CSVs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    feature_path = Path(args.features)
    if not feature_path.is_file():
        LOG.error("Feature file not found: %s", feature_path)
        return 2

    features = pd.read_csv(feature_path, low_memory=False)
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(features.columns))
    if missing:
        LOG.error("Feature file is missing required columns: %s", ", ".join(missing))
        return 2

    features["feature_season_key"] = features["season"].map(normalise_season)
    features["feature_home_key"] = features["home_team"].map(team_key)
    features["feature_away_key"] = features["away_team"].map(team_key)
    features["feature_kickoff_utc"] = pd.to_datetime(features["kickoff_utc"], utc=True, errors="coerce")
    features["lineup_strength_features_ready"] = parse_flag(features["lineup_strength_features_ready"])
    features["lineup_delta_features_ready"] = parse_flag(features["lineup_delta_features_ready"])

    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as connection:
            matches = pd.read_sql_query(MATCHES_SQL, connection)
    except Exception as exc:
        LOG.exception("Unable to read finished match outcomes from PostgreSQL: %s", exc)
        return 1
    finally:
        engine.dispose()

    if matches.empty:
        LOG.error("No completed database matches found for the target seasons.")
        return 1

    matches["database_season_key"] = matches["season"].map(normalise_season)
    matches["database_home_key"] = matches["database_home_team"].map(team_key)
    matches["database_away_key"] = matches["database_away_team"].map(team_key)
    matches["database_kickoff_utc"] = pd.to_datetime(matches["database_kickoff_utc"], utc=True, errors="coerce")
    matches["actual_outcome"] = [outcome_from_goals(h, a) for h, a in zip(matches["home_goals"], matches["away_goals"])]

    feature_keys = ["feature_season_key", "feature_home_key", "feature_away_key"]
    db_keys = ["database_season_key", "database_home_key", "database_away_key"]
    if features.duplicated(feature_keys).any():
        examples = features.loc[features.duplicated(feature_keys, keep=False),
                                ["season", "home_team", "away_team", "fixture_id"]].head(10)
        LOG.error("Feature data has duplicate season/home/away keys. Examples:\n%s", examples.to_string(index=False))
        return 1
    if matches.duplicated(db_keys).any():
        examples = matches.loc[matches.duplicated(db_keys, keep=False),
                               ["season", "database_home_team", "database_away_team", "match_id"]].head(10)
        LOG.error("Database has duplicate season/home/away keys. Examples:\n%s", examples.to_string(index=False))
        return 1

    feature_join = features.merge(
        matches,
        left_on=feature_keys,
        right_on=db_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_db"),
    )
    unmatched_features = feature_join[feature_join["_merge"] != "both"].copy()
    matched_feature_keys = feature_join.loc[feature_join["_merge"] == "both", feature_keys].drop_duplicates()
    feature_key_index = set(map(tuple, matched_feature_keys.to_numpy()))
    db_key_rows = list(map(tuple, matches[db_keys].to_numpy()))
    unmatched_db_mask = [key not in feature_key_index for key in db_key_rows]
    unmatched_db = matches.loc[unmatched_db_mask].copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    feature_join = feature_join.drop(columns=["_merge"])
    feature_join.to_csv(output, index=False)
    unmatched_features.to_csv(report_dir / "unmatched_feature_rows.csv", index=False)
    unmatched_db.to_csv(report_dir / "unmatched_database_matches.csv", index=False)

    matched = int(feature_join["match_id"].notna().sum())
    unique_seasons = sorted(s for s in features["feature_season_key"].dropna().unique())
    print("PLAYER MODEL DATA JOIN (read-only database access)")
    print(f"Feature rows: {len(features):,}")
    print(f"Completed database matches retrieved: {len(matches):,}")
    print(f"Successfully matched feature rows to actual results: {matched:,}")
    print(f"Unmatched feature rows: {len(unmatched_features):,}")
    print(f"Database matches without a feature row: {len(unmatched_db):,}")
    print(f"Seasons in feature file: {', '.join(unique_seasons)}")
    if matched:
        print("\nMatched fixtures by season:")
        print(feature_join[feature_join["match_id"].notna()].groupby("feature_season_key")["match_id"].nunique().to_string())
        ready = feature_join[feature_join["match_id"].notna()]
        print(f"\nMatched fixtures with valid full-XI strength features: {int(ready['lineup_strength_features_ready'].sum())}")
        print(f"Matched fixtures with recent-lineup delta features: {int(ready['lineup_delta_features_ready'].sum())}")
    print(f"\nJoined dataset: {output}")
    print(f"Join audit reports: {report_dir}")

    if len(unmatched_features) or len(unmatched_db):
        print("\nWARNING: some rows did not match. Inspect both unmatched CSVs before model fitting.")
        print("Do not proceed to model training until the unmatched reasons are understood.")
    else:
        print("\nJoin coverage is complete for the queried finished matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())