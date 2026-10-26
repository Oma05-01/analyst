"""
Score locked predictions against finished matches.

Run from the repository root:   python -m src.score_predictions

Scores are stored at full precision (round only when displaying). Every
model_version, including the baseline, goes through the same code.
"""
import logging
import sys

import pandas as pd
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine
from src.metrics import (calculate_brier_score, calculate_log_loss, calculate_rps,
                         outcome_from_goals)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UNSCORED_SQL = text("""
    SELECT p.id AS prediction_id, p.model_version,
           p.p_home, p.p_draw, p.p_away,
           m.home_goals, m.away_goals
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    LEFT JOIN prediction_scores ps ON ps.prediction_id = p.id
    WHERE m.status = 'FINISHED'
      AND m.home_goals IS NOT NULL
      AND m.away_goals IS NOT NULL
      AND ps.prediction_id IS NULL
""")

INSERT_SCORE_SQL = text("""
    INSERT INTO prediction_scores (prediction_id, outcome, brier, rps, log_loss)
    VALUES (:prediction_id, :outcome, :brier, :rps, :log_loss)
    ON CONFLICT (prediction_id) DO NOTHING
""")

SUMMARY_SQL = text("""
    SELECT p.model_version,
           COUNT(*)          AS n,
           AVG(ps.brier)     AS brier,
           AVG(ps.rps)       AS rps,
           AVG(ps.log_loss)  AS log_loss
    FROM prediction_scores ps
    JOIN predictions p ON p.id = ps.prediction_id
    GROUP BY p.model_version
    ORDER BY p.model_version
""")


def score_unscored_predictions(engine):
    unscored = pd.read_sql(UNSCORED_SQL, engine)
    if unscored.empty:
        logging.info("No unscored predictions found.")
        return 0

    rows, bad = [], 0
    for row in unscored.itertuples(index=False):
        try:
            outcome = outcome_from_goals(row.home_goals, row.away_goals)
            p_h, p_d, p_a = float(row.p_home), float(row.p_draw), float(row.p_away)
            rows.append({
                "prediction_id": int(row.prediction_id),
                "outcome": outcome,
                "brier": calculate_brier_score(p_h, p_d, p_a, outcome),
                "rps": calculate_rps(p_h, p_d, p_a, outcome),
                "log_loss": calculate_log_loss(p_h, p_d, p_a, outcome),
            })
        except ValueError:
            bad += 1
            logging.exception("Could not score prediction_id=%s", row.prediction_id)

    if rows:
        with engine.begin() as conn:          # one atomic transaction
            conn.execute(INSERT_SCORE_SQL, rows)
    logging.info("Scored %d predictions (%d could not be scored).", len(rows), bad)
    return bad


def print_summary(engine):
    summary = pd.read_sql(SUMMARY_SQL, engine)
    if summary.empty:
        return
    logging.info("Mean scores by model (lower is better; Brier is summed over 3 classes):\n%s",
                 summary.round(4).to_string(index=False))


def main():
    engine = get_sqlalchemy_engine()
    try:
        bad = score_unscored_predictions(engine)
        print_summary(engine)
        return 1 if bad else 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())