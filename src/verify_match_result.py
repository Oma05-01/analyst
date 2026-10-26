#!/usr/bin/env python3
"""Read-only verification of a match result and its saved prediction scores.

Run from the repository root, for example:
    python -m src.verify_match_result
    python -m src.verify_match_result --home Ipswich --away "Nottingham Forest" --date 2026-10-23

The script does not insert, update, or delete any database records.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

from src.db import get_db_connection


QUERY = """
SELECT
    m.id AS match_id,
    m.season,
    m.kickoff_utc,
    m.status,
    ht.name AS home_team,
    awt.name AS away_team,
    m.home_goals,
    m.away_goals,
    p.id AS prediction_id,
    p.model_version,
    p.p_home,
    p.p_draw,
    p.p_away,
    p.locked_at,
    p.train_cutoff,
    p.n_train,
    ps.outcome AS scored_outcome,
    ps.brier,
    ps.rps,
    ps.log_loss,
    (
        SELECT COUNT(*)
        FROM predictions p_count
        WHERE p_count.match_id = m.id
    ) AS prediction_count,
    (
        SELECT COUNT(*)
        FROM prediction_scores ps_count
        JOIN predictions p_count
          ON p_count.id = ps_count.prediction_id
        WHERE p_count.match_id = m.id
    ) AS scored_prediction_count
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams awt ON awt.id = m.away_team_id
LEFT JOIN predictions p ON p.match_id = m.id
LEFT JOIN prediction_scores ps ON ps.prediction_id = p.id
WHERE ht.name ILIKE %s
  AND awt.name ILIKE %s
  AND m.kickoff_utc >= %s
  AND m.kickoff_utc < %s
ORDER BY p.model_version NULLS LAST;
"""


def fmt_datetime(value):
    return value.isoformat(sep=" ") if value is not None else "not available"


def fmt_number(value, digits=4):
    return f"{float(value):.{digits}f}" if value is not None else "not available"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="Ipswich", help="Home-team name or substring (default: Ipswich)")
    parser.add_argument("--away", default="Nottingham Forest", help="Away-team name or substring (default: Nottingham Forest)")
    parser.add_argument("--date", default="2026-10-23", help="UTC calendar date YYYY-MM-DD (default: 2026-10-23)")
    args = parser.parse_args()

    try:
        match_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        parser.error("--date must use YYYY-MM-DD format")

    start_utc = datetime.combine(match_date, datetime.min.time(), tzinfo=timezone.utc)
    end_utc = start_utc + timedelta(days=1)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                QUERY,
                (f"%{args.home}%", f"%{args.away}%", start_utc, end_utc),
            )
            rows = cur.fetchall()
        # Explicit rollback makes the read-only intent clear even if the driver
        # opened a normal transaction. No mutation statements are issued.
        conn.rollback()

        print("MATCH RESULT VERIFICATION (READ-ONLY)")
        print(f"Search: {args.home} (home) vs {args.away} (away), UTC date {args.date}")

        if not rows:
            print("\nNo matching fixture found for those team names and UTC date.")
            print("Check the team names, match date, and whether the fixture exists locally.")
            return 1

        first = rows[0]
        (match_id, season, kickoff, status, home_team, away_team, home_goals,
         away_goals, _prediction_id, _model_version, _ph, _pd, _pa, _locked_at,
         _train_cutoff, _n_train, _outcome, _brier, _rps, _log_loss,
         prediction_count, scored_prediction_count) = first

        print("\nFIXTURE")
        print(f"Match ID:              {match_id}")
        print(f"Fixture:               {home_team} vs {away_team}")
        print(f"Season:                {season}")
        print(f"Kickoff (UTC):         {fmt_datetime(kickoff)}")
        print(f"Status:                {status}")
        if home_goals is None or away_goals is None:
            print("Stored result:         not available")
        else:
            print(f"Stored result:         {home_goals}-{away_goals}")
        print(f"Prediction records:    {prediction_count}")
        print(f"Scored predictions:    {scored_prediction_count}")

        print("\nSAVED PREDICTIONS")
        predictions = [row for row in rows if row[8] is not None]
        if not predictions:
            print("No saved predictions found for this fixture.")
        else:
            for row in predictions:
                (match_id, season, kickoff, status, home_team, away_team, home_goals,
                 away_goals, prediction_id, model_version, p_home, p_draw, p_away,
                 locked_at, train_cutoff, n_train, scored_outcome, brier, rps,
                 log_loss, _prediction_count, _scored_count) = row
                print("\n" + "-" * 68)
                print(f"Prediction ID: {prediction_id}")
                print(f"Model version: {model_version}")
                print(f"Locked at (UTC): {fmt_datetime(locked_at)}")
                print(f"Training cutoff: {fmt_datetime(train_cutoff)}")
                print(f"Training matches: {n_train if n_train is not None else 'not available'}")
                print(
                    "Probabilities:       "
                    f"home={fmt_number(p_home)}  draw={fmt_number(p_draw)}  away={fmt_number(p_away)}"
                )
                if scored_outcome is None:
                    print("Scoring:              not scored yet")
                else:
                    print(f"Scored outcome:       {scored_outcome}")
                    print(
                        "Scores:               "
                        f"Brier={fmt_number(brier)}  RPS={fmt_number(rps)}  "
                        f"Log loss={fmt_number(log_loss)}"
                    )

        if status == "FINISHED" and home_goals is not None and away_goals is not None:
            if scored_prediction_count == prediction_count:
                print("\nCHECK: Fixture is FINISHED and all saved predictions have scores.")
            else:
                print("\nCHECK: Fixture is FINISHED, but some predictions remain unscored. Run:")
                print("  python -m src.score_predictions")
        elif status != "FINISHED" or home_goals is None or away_goals is None:
            print("\nCHECK: Fixture is not fully settled in the local database.")

        return 0
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        print(f"Database verification failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
