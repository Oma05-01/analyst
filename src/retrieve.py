"""Retrieve saved predictions for a specific Premier League fixture.

Run from the repository root, with the project environment active:
    python -m src.retrieve_match_prediction

Defaults to Ipswich Town vs Nottingham Forest on 2026-10-23. Override with:
    python -m src.retrieve_match_prediction --home Ipswich --away "Nottingham Forest" --date 2026-10-23

This script is read-only: it does not create, update, or delete database records.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine

MATCH_SQL = text("""
    SELECT
        m.id AS match_id,
        m.season,
        m.matchweek,
        m.kickoff_utc,
        m.status AS match_status,
        m.home_goals,
        m.away_goals,
        ht.name AS home_team,
        awt.name AS away_team,
        p.id AS prediction_id,
        p.model_version,
        p.p_home,
        p.p_draw,
        p.p_away,
        p.expected_home_goals,
        p.expected_away_goals,
        p.scoreline_matrix,
        p.payload_hash,
        p.n_train,
        p.train_cutoff,
        p.locked_at,
        ps.brier,
        ps.rps,
        ps.log_loss,
        mp.text AS analyst_preview,
        mp.llm_model AS analyst_model
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams awt ON awt.id = m.away_team_id
    LEFT JOIN predictions p ON p.match_id = m.id
    LEFT JOIN prediction_scores ps ON ps.prediction_id = p.id
    LEFT JOIN match_previews mp
           ON mp.match_id = p.match_id
          AND mp.model_version = p.model_version
    WHERE ht.name ILIKE '%' || :home || '%'
      AND awt.name ILIKE '%' || :away || '%'
      AND m.kickoff_utc >= (CAST(:match_date AS date)::timestamp AT TIME ZONE 'UTC')
      AND m.kickoff_utc < ((CAST(:match_date AS date) + INTERVAL '1 day') AT TIME ZONE 'UTC')
    ORDER BY m.kickoff_utc, p.model_version NULLS LAST
""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a fixture and print any saved predictions, scores, and analyst previews."
    )
    parser.add_argument("--home", default="Ipswich", help="Home-team name fragment (default: Ipswich)")
    parser.add_argument("--away", default="Nottingham Forest", help="Away-team name fragment (default: Nottingham Forest)")
    parser.add_argument("--date", dest="match_date", default="2026-10-23",
                        help="Fixture date in YYYY-MM-DD format (default: 2026-10-23; date range is UTC)")
    parser.add_argument("--show-matrix", action="store_true",
                        help="Print the complete scoreline-probability matrix for each saved prediction")
    return parser.parse_args()


def display_value(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "not available"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def decode_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = json.loads(value)
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
            return None
        return matrix
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def print_matrix_summary(value: Any, show_matrix: bool) -> None:
    matrix = decode_matrix(value)
    if matrix is None:
        print("  Scoreline matrix: unavailable or invalid")
        return

    flat_order = np.argsort(matrix.ravel())[::-1]
    print("  Most likely exact scorelines:")
    shown = 0
    for flat_index in flat_order:
        probability = float(matrix.ravel()[flat_index])
        if probability <= 0 or shown >= 8:
            break
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        print(f"    {home_goals}-{away_goals}: {probability:.2%}")
        shown += 1

    if show_matrix:
        print("  Full scoreline matrix (rows=home goals, columns=away goals; values in %):")
        print(pd.DataFrame(matrix * 100).round(3).to_string(index=True, header=True))


def report(df: pd.DataFrame, args: argparse.Namespace) -> int:
    if df.empty:
        print("No matching fixture found for the given teams and UTC date.")
        print("Check the team-name fragments and date, or try a neighbouring date if the kickoff crosses UTC midnight.")
        return 1

    fixtures = df[["match_id", "season", "matchweek", "kickoff_utc", "match_status",
                    "home_team", "away_team", "home_goals", "away_goals"]].drop_duplicates()
    print("FIXTURE LOOKUP (read-only)\n")
    for fixture in fixtures.itertuples(index=False):
        print(f"Match ID: {fixture.match_id}")
        print(f"Fixture: {fixture.home_team} vs {fixture.away_team}")
        print(f"Season / matchweek: {fixture.season} / {fixture.matchweek}")
        print(f"Kickoff (UTC): {fixture.kickoff_utc}")
        print(f"Status: {fixture.match_status}")
        if pd.notna(fixture.home_goals) and pd.notna(fixture.away_goals):
            print(f"Actual score currently stored: {int(fixture.home_goals)}-{int(fixture.away_goals)}")
        else:
            print("Actual score currently stored: not available")

        predictions = df[df["match_id"] == fixture.match_id]
        predictions = predictions[predictions["prediction_id"].notna()]
        if predictions.empty:
            print("\nNo saved prediction rows found for this fixture.")
            print("\n" + "-" * 72 + "\n")
            continue

        for row in predictions.itertuples(index=False):
            print("\n" + "=" * 72)
            print(f"Prediction ID: {int(row.prediction_id)}")
            print(f"Model version: {row.model_version}")
            print(f"Locked at: {row.locked_at}")
            print(f"Training cutoff: {row.train_cutoff}")
            print(f"Training matches: {row.n_train}")
            print("1X2 probabilities:")
            print(f"  Home win: {display_value(row.p_home)}")
            print(f"  Draw:     {display_value(row.p_draw)}")
            print(f"  Away win: {display_value(row.p_away)}")
            print("Expected goals:")
            print(f"  Home: {display_value(row.expected_home_goals, 3)}")
            print(f"  Away: {display_value(row.expected_away_goals, 3)}")
            print_matrix_summary(row.scoreline_matrix, args.show_matrix)

            scores = (row.brier, row.rps, row.log_loss)
            if all(pd.notna(score) for score in scores):
                print("Stored evaluation scores (lower is better):")
                print(f"  Brier:   {float(row.brier):.6f}")
                print(f"  RPS:     {float(row.rps):.6f}")
                print(f"  Log loss:{float(row.log_loss):.6f}")
            else:
                print("Evaluation scores: not recorded yet (the scorer may run after the match is settled).")

            if row.analyst_preview and str(row.analyst_preview).strip():
                print(f"\nAnalyst preview ({row.analyst_model or 'model not recorded'}):")
                print(str(row.analyst_preview).strip())
            else:
                print("\nAnalyst preview: not saved")

        print("\n" + "-" * 72 + "\n")

    return 0


def main() -> int:
    args = parse_args()
    try:
        date.fromisoformat(args.match_date)
    except ValueError:
        print("Invalid --date value. Use YYYY-MM-DD, for example 2026-10-23.", file=sys.stderr)
        return 2

    engine = get_sqlalchemy_engine()
    try:
        df = pd.read_sql_query(
            MATCH_SQL,
            engine,
            params={"home": args.home.strip(), "away": args.away.strip(),
                    "match_date": args.match_date},
        )
        return report(df, args)
    except Exception as exc:
        print(f"Could not retrieve prediction: {exc}", file=sys.stderr)
        print("Check that DATABASE_URL is set and the project migrations have been applied.", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
