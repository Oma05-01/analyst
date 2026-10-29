#!/usr/bin/env python3
"""Build leakage-aware recent-form and head-to-head features for PL fixtures.

Reads match data from PostgreSQL and writes a CSV only; it does not update the DB.
Run from the repository root:
    python -m src.build_match_context_features
    python -m src.build_match_context_features --output data/processed/match_context_features.csv

Feature timing rule: all same-UTC-calendar-day matches are excluded from the
history used for a target fixture. This is deliberately conservative because
historical kickoff times may not preserve reliable within-day ordering.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine

LOG = logging.getLogger("build_match_context_features")

FIXTURES_SQL = text("""
    SELECT m.id AS match_id,
           m.season,
           m.matchweek,
           m.kickoff_utc,
           m.status,
           m.home_team_id,
           m.away_team_id,
           ht.name AS home_team,
           awt.name AS away_team,
           m.home_goals,
           m.away_goals
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams awt ON awt.id = m.away_team_id
    WHERE m.kickoff_utc IS NOT NULL
    ORDER BY m.kickoff_utc, m.id
""")

FORM_WINDOW = 5
H2H_WINDOW = 5


def _valid_finished(row: Any) -> bool:
    status = str(row.status).strip().upper()
    try:
        scores_present = pd.notna(row.home_goals) and pd.notna(row.away_goals)
    except AttributeError:
        scores_present = pd.notna(row["home_goals"]) and pd.notna(row["away_goals"])
    return status == "FINISHED" and bool(scores_present)


def _form_features(history: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    recent = history[-FORM_WINDOW:]
    n = len(recent)
    if n == 0:
        return {
            f"{prefix}_matches": 0,
            f"{prefix}_wins": 0,
            f"{prefix}_draws": 0,
            f"{prefix}_losses": 0,
            f"{prefix}_points": 0,
            f"{prefix}_points_per_game": None,
            f"{prefix}_goal_difference": 0,
            f"{prefix}_goal_difference_per_game": None,
            f"{prefix}_goals_for_per_game": None,
            f"{prefix}_goals_against_per_game": None,
        }

    wins = sum(int(item["points"]) == 3 for item in recent)
    draws = sum(int(item["points"]) == 1 for item in recent)
    losses = n - wins - draws
    points = sum(int(item["points"]) for item in recent)
    goals_for = sum(int(item["goals_for"]) for item in recent)
    goals_against = sum(int(item["goals_against"]) for item in recent)
    goal_difference = goals_for - goals_against
    return {
        f"{prefix}_matches": n,
        f"{prefix}_wins": wins,
        f"{prefix}_draws": draws,
        f"{prefix}_losses": losses,
        f"{prefix}_points": points,
        f"{prefix}_points_per_game": points / n,
        f"{prefix}_goal_difference": goal_difference,
        f"{prefix}_goal_difference_per_game": goal_difference / n,
        f"{prefix}_goals_for_per_game": goals_for / n,
        f"{prefix}_goals_against_per_game": goals_against / n,
    }


def _h2h_features(
    history: list[dict[str, Any]], target_home_id: int, target_away_id: int
) -> dict[str, Any]:
    recent = history[-H2H_WINDOW:]
    n = len(recent)
    home_wins = draws = away_wins = home_goals = away_goals = home_points = 0

    for item in recent:
        # Re-orient each past meeting to the home/away orientation of the target.
        if int(item["home_team_id"]) == target_home_id:
            target_home_goals = int(item["home_goals"])
            target_away_goals = int(item["away_goals"])
        else:
            target_home_goals = int(item["away_goals"])
            target_away_goals = int(item["home_goals"])

        home_goals += target_home_goals
        away_goals += target_away_goals
        if target_home_goals > target_away_goals:
            home_wins += 1
            home_points += 3
        elif target_home_goals == target_away_goals:
            draws += 1
            home_points += 1
        else:
            away_wins += 1

    return {
        "h2h_matches": n,
        "h2h_home_wins": home_wins,
        "h2h_draws": draws,
        "h2h_away_wins": away_wins,
        "h2h_home_points_per_game": (home_points / n) if n else None,
        "h2h_goal_difference_per_game": ((home_goals - away_goals) / n) if n else None,
        "h2h_home_goals_per_game": (home_goals / n) if n else None,
        "h2h_away_goals_per_game": (away_goals / n) if n else None,
    }


def build_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id", "season", "matchweek", "kickoff_utc", "status",
        "home_team_id", "away_team_id", "home_team", "away_team",
        "home_goals", "away_goals",
    }
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"Fixture data is missing required columns: {missing}")

    df = fixtures.copy()
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["match_id", "kickoff_utc", "home_team_id", "away_team_id"]).copy()
    df["match_id"] = df["match_id"].astype(int)
    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)
    df["_utc_day"] = df["kickoff_utc"].dt.floor("D")
    df = df.sort_values(["_utc_day", "kickoff_utc", "match_id"]).reset_index(drop=True)

    # Keep history by team and by unordered pair of teams. Only completed matches
    # from strictly earlier UTC calendar dates are placed into these collections.
    team_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pair_history: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    out_by_match_id: dict[int, dict[str, Any]] = {}

    for day, day_rows in df.groupby("_utc_day", sort=True):
        # First calculate all features for the day before adding any results from it.
        for row in day_rows.itertuples(index=False):
            home_id = int(row.home_team_id)
            away_id = int(row.away_team_id)
            h_hist = team_history[home_id]
            a_hist = team_history[away_id]
            pair_key = (min(home_id, away_id), max(home_id, away_id))
            h2h_hist = pair_history[pair_key]

            features: dict[str, Any] = {
                "match_id": int(row.match_id),
                "season": row.season,
                "matchweek": row.matchweek,
                "kickoff_utc": row.kickoff_utc.isoformat(),
                "feature_cutoff_utc_date": day.date().isoformat(),
                "status_at_build": row.status,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "history_rule": "finished_matches_before_target_utc_date_only",
            }
            features.update(_form_features(h_hist, "home_form"))
            features.update(_form_features(a_hist, "away_form"))
            features.update(_h2h_features(h2h_hist, home_id, away_id))
            out_by_match_id[int(row.match_id)] = features

        # Only after every fixture on this date has been featured do its finished
        # results become available to fixtures on subsequent UTC dates.
        for row in day_rows.itertuples(index=False):
            if not _valid_finished(row):
                continue
            home_id = int(row.home_team_id)
            away_id = int(row.away_team_id)
            hg = int(row.home_goals)
            ag = int(row.away_goals)
            if hg > ag:
                home_points, away_points = 3, 0
            elif hg < ag:
                home_points, away_points = 0, 3
            else:
                home_points = away_points = 1

            team_history[home_id].append({
                "points": home_points, "goals_for": hg, "goals_against": ag,
                "match_id": int(row.match_id), "utc_day": day,
            })
            team_history[away_id].append({
                "points": away_points, "goals_for": ag, "goals_against": hg,
                "match_id": int(row.match_id), "utc_day": day,
            })
            pair_key = (min(home_id, away_id), max(home_id, away_id))
            pair_history[pair_key].append({
                "home_team_id": home_id, "away_team_id": away_id,
                "home_goals": hg, "away_goals": ag,
                "match_id": int(row.match_id), "utc_day": day,
            })

    result = pd.DataFrame(out_by_match_id.values())
    if result["match_id"].duplicated().any():
        raise ValueError("Feature builder produced duplicate match_id values")
    return result.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="data/processed/match_context_features.csv",
        help="Path for the generated context-feature CSV",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as connection:
            fixtures = pd.read_sql_query(FIXTURES_SQL, connection)
    except Exception:
        LOG.exception("Unable to read fixtures from PostgreSQL")
        return 1
    finally:
        engine.dispose()

    if fixtures.empty:
        LOG.error("No fixtures found in the database")
        return 1

    try:
        output_df = build_features(fixtures)
    except Exception:
        LOG.exception("Failed to calculate match-context features")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format="%.8g")

    finished = fixtures["status"].astype(str).str.upper().eq("FINISHED")
    score_ready = fixtures["home_goals"].notna() & fixtures["away_goals"].notna()
    print("MATCH CONTEXT FEATURES (read-only database access)")
    print(f"Fixture rows retrieved: {len(fixtures):,}")
    print(f"Finished matches with scores: {int((finished & score_ready).sum()):,}")
    print(f"Feature rows written: {len(output_df):,}")
    print(f"Form window: last {FORM_WINDOW} completed matches per team")
    print(f"Head-to-head window: last {H2H_WINDOW} meetings")
    print("Leakage guard: results from the same UTC calendar date are excluded")
    print(f"Output: {output_path}")

    for col in ("home_form_matches", "away_form_matches", "h2h_matches"):
        if col in output_df:
            print(f"Rows with {col} > 0: {int((output_df[col] > 0).sum()):,}")
    print("\nReview the CSV before connecting these features to a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
