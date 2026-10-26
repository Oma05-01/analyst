"""Date-batched, expanding-window backtest for the Premier League model.

Run from the repository root, for example:
    python -m src.evaluate_models --season 2024-2025

This script is read-only with respect to the database. It does not insert,
update, or delete production predictions or score records.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine
from src.metrics import (
    calculate_brier_score,
    calculate_log_loss,
    calculate_rps,
    outcome_from_goals,
)
from src.models.dixon_coles import (
    DixonColesModel,
    UnknownTeamError,
    outcome_probs,
    score_matrix,
)

# Keep these settings aligned with src.generate_predictions.
MODEL_VERSION = "dixon_coles_v2"
BASELINE_VERSION = "baseline_poisson_v1"
TIME_DECAY = 0.001
RIDGE = 1.0
MAX_GOALS = 9
MIN_TRAIN_MATCHES = 200

MATCHES_SQL = text("""
    SELECT m.id AS match_id,
           m.season,
           m.matchweek,
           m.kickoff_utc,
           m.home_team_id,
           m.away_team_id,
           ht.name AS home_name,
           at.name AS away_name,
           m.home_goals,
           m.away_goals
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams at ON at.id = m.away_team_id
    WHERE m.status = 'FINISHED'
      AND m.home_goals IS NOT NULL
      AND m.away_goals IS NOT NULL
    ORDER BY m.kickoff_utc ASC, m.id ASC
""")


def league_average_rates(
    training_df: pd.DataFrame,
    reference_date: pd.Timestamp,
    time_decay: float,
) -> tuple[float, float]:
    """Calculate decay-weighted league-average home and away goals."""
    kickoff = pd.to_datetime(training_df["kickoff_utc"], utc=True)
    days_old = (
        (reference_date - kickoff).dt.total_seconds() / 86400.0
    ).clip(lower=0.0).to_numpy()
    weights = np.exp(-time_decay * days_old)
    home_goals = training_df["home_goals"].to_numpy(dtype=float)
    away_goals = training_df["away_goals"].to_numpy(dtype=float)

    lam = float(np.average(home_goals, weights=weights))
    mu = float(np.average(away_goals, weights=weights))
    if not np.isfinite([lam, mu]).all() or lam <= 0 or mu <= 0:
        raise ValueError(f"Invalid league-average rates: lambda={lam}, mu={mu}")
    return lam, mu


def validate_distribution(matrix: np.ndarray, probs: tuple[float, float, float]) -> None:
    """Fail loudly rather than silently scoring malformed model probabilities."""
    matrix = np.asarray(matrix, dtype=float)
    p = np.asarray(probs, dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(p).all():
        raise ValueError("Prediction contains NaN or infinite values")
    if (matrix < 0).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Prediction contains probabilities outside [0, 1]")
    if abs(float(matrix.sum()) - 1.0) > 1e-8:
        raise ValueError(f"Score matrix sums to {matrix.sum()}, expected 1")
    if abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(f"1X2 probabilities sum to {p.sum()}, expected 1")
    derived = outcome_probs(matrix)
    if any(abs(float(a) - float(b)) > 1e-8 for a, b in zip(derived, p)):
        raise ValueError("1X2 probabilities disagree with the score matrix")


def score_probs(
    probs: tuple[float, float, float], outcome: str
) -> dict[str, float]:
    p_home, p_draw, p_away = map(float, probs)
    scores = {
        "brier": calculate_brier_score(p_home, p_draw, p_away, outcome),
        "rps": calculate_rps(p_home, p_draw, p_away, outcome),
        "log_loss": calculate_log_loss(p_home, p_draw, p_away, outcome),
    }
    if not np.isfinite(list(scores.values())).all():
        raise ValueError(f"Scoring function returned a non-finite value: {scores}")
    return scores


def load_matches(engine) -> pd.DataFrame:
    matches = pd.read_sql(MATCHES_SQL, engine)
    if matches.empty:
        raise RuntimeError("No completed matches with scores were found in the database")

    matches["kickoff_utc"] = pd.to_datetime(
        matches["kickoff_utc"], utc=True, errors="coerce"
    )
    matches = matches.dropna(
        subset=[
            "kickoff_utc", "home_team_id", "away_team_id",
            "home_goals", "away_goals", "season",
        ]
    ).copy()
    # Group by UTC calendar date. Historical kickoff times can be artificial,
    # so same-date fixtures must be held out together regardless of their time.
    matches["_match_date"] = matches["kickoff_utc"].dt.date
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    return matches.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def run_backtest(
    matches: pd.DataFrame,
    season: str,
    min_train_matches: int = MIN_TRAIN_MATCHES,
    max_dates: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Backtest one season, retraining only on matches from earlier dates.

    All fixtures on a target date share the same model and baseline trained on
    strictly earlier dates. The function performs no database writes.
    """
    evaluation = matches[matches["season"].astype(str) == str(season)].copy()
    if evaluation.empty:
        available = sorted(matches["season"].astype(str).unique())
        raise ValueError(
            f"No completed matches found for season {season!r}. "
            f"Available completed seasons: {available}"
        )

    dates = sorted(evaluation["_match_date"].unique())
    if max_dates is not None:
        if max_dates < 1:
            raise ValueError("--max-dates must be at least 1")
        dates = dates[:max_dates]

    records: list[dict] = []
    skipped_unknown_team = 0
    skipped_insufficient_history = 0

    for eval_date in dates:
        # Critical leakage guard: *all* fixtures dated eval_date are excluded.
        train = matches[matches["_match_date"] < eval_date].copy()
        test_day = evaluation[evaluation["_match_date"] == eval_date]

        if len(train) < min_train_matches:
            skipped_insufficient_history += len(test_day)
            logging.info(
                "Skipping %s: only %d earlier matches (minimum %d)",
                eval_date, len(train), min_train_matches,
            )
            continue

        reference_date = pd.Timestamp(eval_date).tz_localize("UTC")
        train_for_model = train.drop(columns=["_match_date"])

        # Deliberately let fitting/validation errors fail the backtest. A partial
        # result should not silently look like a successful model evaluation.
        model = DixonColesModel(time_decay=TIME_DECAY, ridge=RIDGE)
        model.fit(train_for_model, reference_date=reference_date)

        base_lam, base_mu = league_average_rates(train, reference_date, TIME_DECAY)
        base_matrix = score_matrix(base_lam, base_mu, 0.0, MAX_GOALS)
        base_probs = outcome_probs(base_matrix)
        validate_distribution(base_matrix, base_probs)

        logging.info(
            "Evaluating %s: %d training matches, %d fixtures",
            eval_date, len(train), len(test_day),
        )

        for match in test_day.itertuples(index=False):
            try:
                p_home, p_draw, p_away, lam, mu, matrix = model.predict_probs(
                    int(match.home_team_id), int(match.away_team_id), MAX_GOALS
                )
            except UnknownTeamError as exc:
                skipped_unknown_team += 1
                logging.warning(
                    "Skipping match %s (%s vs %s): unknown team %s",
                    match.match_id, match.home_name, match.away_name, exc,
                )
                continue

            dc_probs = (float(p_home), float(p_draw), float(p_away))
            validate_distribution(matrix, dc_probs)
            outcome = outcome_from_goals(match.home_goals, match.away_goals)
            dc_scores = score_probs(dc_probs, outcome)
            baseline_scores = score_probs(base_probs, outcome)

            records.append({
                "match_id": int(match.match_id),
                "season": str(match.season),
                "match_date_utc": str(eval_date),
                "matchweek": match.matchweek,
                "home_name": match.home_name,
                "away_name": match.away_name,
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
                "actual_outcome": outcome,
                "n_train": int(len(train)),
                "train_cutoff_date_exclusive": str(eval_date),
                "dc_p_home": dc_probs[0],
                "dc_p_draw": dc_probs[1],
                "dc_p_away": dc_probs[2],
                "dc_expected_home_goals": float(lam),
                "dc_expected_away_goals": float(mu),
                "dc_brier": dc_scores["brier"],
                "dc_rps": dc_scores["rps"],
                "dc_log_loss": dc_scores["log_loss"],
                "baseline_p_home": float(base_probs[0]),
                "baseline_p_draw": float(base_probs[1]),
                "baseline_p_away": float(base_probs[2]),
                "baseline_expected_home_goals": base_lam,
                "baseline_expected_away_goals": base_mu,
                "baseline_brier": baseline_scores["brier"],
                "baseline_rps": baseline_scores["rps"],
                "baseline_log_loss": baseline_scores["log_loss"],
            })

    results = pd.DataFrame.from_records(records)
    metadata = {
        "season": season,
        "dates_considered": len(dates),
        "matches_evaluated": len(results),
        "unknown_team_matches_skipped": skipped_unknown_team,
        "insufficient_history_matches_skipped": skipped_insufficient_history,
        "min_train_matches": min_train_matches,
        "max_dates_limit": max_dates,
    }
    if results.empty:
        raise RuntimeError(
            "No matches were evaluated. Check season, training history, and team coverage. "
            f"Details: {metadata}"
        )
    return results, metadata


def print_report(results: pd.DataFrame, metadata: dict) -> None:
    metrics = ["brier", "rps", "log_loss"]
    rows = []
    for model, prefix in ((MODEL_VERSION, "dc"), (BASELINE_VERSION, "baseline")):
        rows.append({
            "model": model,
            "n": len(results),
            **{
                metric: float(results[f"{prefix}_{metric}"].mean())
                for metric in metrics
            },
        })
    summary = pd.DataFrame(rows)

    print("\nEXPANDING-WINDOW BACKTEST")
    print(f"Season under evaluation: {metadata['season']}")
    print(f"Fixtures evaluated (paired): {metadata['matches_evaluated']}")
    print(f"Evaluation dates considered: {metadata['dates_considered']}")
    print(f"Minimum prior training matches: {metadata['min_train_matches']}")
    print(f"Skipped: {metadata['unknown_team_matches_skipped']} unknown-team; "
          f"{metadata['insufficient_history_matches_skipped']} insufficient-history")
    print("\nMean scores (lower is better; Brier is summed over 3 classes):")
    print(summary.round(5).to_string(index=False))

    print("\nPaired difference (Dixon-Coles minus baseline; negative favours Dixon-Coles):")
    for metric in metrics:
        dc_col = f"dc_{metric}"
        base_col = f"baseline_{metric}"
        delta = float((results[dc_col] - results[base_col]).mean())
        print(f"  {metric}: {delta:+.6f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", default="2024-2025",
        help="season label to evaluate (default: 2024-2025)",
    )
    parser.add_argument(
        "--min-train-matches", type=int, default=MIN_TRAIN_MATCHES,
        help=f"minimum earlier matches required before evaluation (default: {MIN_TRAIN_MATCHES})",
    )
    parser.add_argument(
        "--max-dates", type=int, default=None,
        help="optional smoke-test limit on the first N evaluation dates",
    )
    parser.add_argument(
        "--output", default=None,
        help="optional CSV path for per-match paired predictions and scores",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.min_train_matches < 1:
        logging.error("--min-train-matches must be at least 1")
        return 2

    engine = get_sqlalchemy_engine()
    try:
        matches = load_matches(engine)
        results, metadata = run_backtest(
            matches,
            season=args.season,
            min_train_matches=args.min_train_matches,
            max_dates=args.max_dates,
        )
        print_report(results, metadata)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False, float_format="%.10g")
            print(f"\nPer-match results saved to: {output_path}")
        return 0
    except Exception:
        logging.exception("Backtest failed")
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
