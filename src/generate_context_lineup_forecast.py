#!/usr/bin/env python3
"""Generate a separate experimental Dixon-Coles + form + H2H + lineup forecast.

This is an inference companion to evaluate_context_lineup.py. It does not write
or modify PostgreSQL predictions, locked baseline rows, or match previews.
It reads the latest saved lineup snapshot and writes to a separate JSONL store
only when --save is supplied.

Run from the repository root:
    python -m src.build_match_context_features
    python -m src.generate_context_lineup_forecast --match-id 7242 --lineup-type projected
    python -m src.generate_context_lineup_forecast --match-id 7242 --lineup-type projected --save
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.db_engine import get_sqlalchemy_engine
from src.evaluate_match_context import (
    FORM_FEATURE,
    H2H_FEATURE,
    load_match_data as load_context_matches,
    validate_distribution,
)
from src.evaluate_context_lineup import ContextLineupDixonColes
from src.generate_lineup_forecast import (
    LINEUP_STORE,
    PLAYER_FEATURES_PATH,
    PLAYER_HISTORY_PATH,
    PLAYER_MODEL_DATASET_PATH,
    PLAYER_MODEL_METADATA_PATH,
    append_jsonl,
    canonical_hash,
    load_finished_training,
    load_frozen_player_params,
    load_upcoming_fixture,
    make_forecast,
)
from src.models.dixon_coles import outcome_probs

LOG = logging.getLogger("generate_context_lineup_forecast")
MODEL_VERSION = "dixon_coles_form_h2h_player_v1"
BASE_MODEL_VERSION = "dixon_coles_v2"
MAX_GOALS = 9
MIN_TRAIN_MATCHES = 200
CONTEXT_PATH = Path("data/processed/match_context_features.csv")
FORECAST_STORE = Path("data/lineups/context_lineup_forecasts.jsonl")


def _read_target_context(context_path: Path, fixture: pd.Series,
                         target_date: pd.Timestamp) -> tuple[float, float, dict[str, Any]]:
    if not context_path.is_file():
        raise FileNotFoundError(
            f"Context feature file not found: {context_path}. Run "
            "python -m src.build_match_context_features first."
        )
    frame = pd.read_csv(context_path, low_memory=False)
    required = {
        "match_id", "feature_cutoff_utc_date", "home_team_id", "away_team_id",
        "home_form_matches", "home_form_wins", "away_form_matches", "away_form_wins",
        "h2h_matches", "h2h_home_wins", "h2h_away_wins", "history_rule",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Context feature file is missing columns: {missing}")

    frame["match_id"] = pd.to_numeric(frame["match_id"], errors="coerce")
    rows = frame.loc[frame["match_id"].eq(int(fixture["match_id"]))]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one context row for match_id={fixture['match_id']}; found {len(rows)}. "
            "Rebuild match-context features."
        )
    row = rows.iloc[0]
    expected_day = target_date.strftime("%Y-%m-%d")
    if str(row["feature_cutoff_utc_date"]) != expected_day:
        raise ValueError(
            f"Context cutoff date {row['feature_cutoff_utc_date']} does not match "
            f"fixture UTC date {expected_day}. Rebuild match-context features."
        )
    if str(row["history_rule"]) != "finished_matches_before_target_utc_date_only":
        raise ValueError(f"Unexpected context history rule: {row['history_rule']}")
    if int(row["home_team_id"]) != int(fixture["home_team_id"]):
        raise ValueError("Context home team ID does not match the database fixture")
    if int(row["away_team_id"]) != int(fixture["away_team_id"]):
        raise ValueError("Context away team ID does not match the database fixture")

    numeric_names = [
        "home_form_matches", "home_form_wins", "away_form_matches", "away_form_wins",
        "h2h_matches", "h2h_home_wins", "h2h_away_wins",
    ]
    values = {name: pd.to_numeric(row[name], errors="coerce") for name in numeric_names}
    if not all(pd.notna(value) and np.isfinite(float(value)) and float(value) >= 0
               for value in values.values()):
        raise ValueError("Context history counts contain missing, negative or non-finite values")
    if values["home_form_matches"] > 5 or values["away_form_matches"] > 5 or values["h2h_matches"] > 5:
        raise ValueError("Context row exceeds the configured five-match windows")
    if values["home_form_wins"] > values["home_form_matches"]:
        raise ValueError("Home form wins exceed available home form matches")
    if values["away_form_wins"] > values["away_form_matches"]:
        raise ValueError("Away form wins exceed available away form matches")
    if values["h2h_home_wins"] + values["h2h_away_wins"] > values["h2h_matches"]:
        raise ValueError("Head-to-head wins exceed available meetings")

    def win_rate(wins: float, matches: float) -> float:
        # Same neutral prior used by the historical evaluator when no matches exist.
        return float(wins / matches) if matches > 0 else 1.0 / 3.0

    form_diff = win_rate(values["home_form_wins"], values["home_form_matches"]) - win_rate(
        values["away_form_wins"], values["away_form_matches"]
    )
    h2h_n = float(values["h2h_matches"])
    h2h_balance = (
        (float(values["h2h_home_wins"]) - float(values["h2h_away_wins"])) / h2h_n
        if h2h_n > 0 else 0.0
    )
    detail = {
        "feature_cutoff_utc_date": expected_day,
        "history_rule": str(row["history_rule"]),
        "home_form_matches": int(values["home_form_matches"]),
        "home_form_wins": int(values["home_form_wins"]),
        "away_form_matches": int(values["away_form_matches"]),
        "away_form_wins": int(values["away_form_wins"]),
        "h2h_matches": int(values["h2h_matches"]),
        "h2h_home_wins": int(values["h2h_home_wins"]),
        "h2h_away_wins": int(values["h2h_away_wins"]),
        FORM_FEATURE: float(form_diff),
        H2H_FEATURE: float(h2h_balance),
    }
    return float(form_diff), float(h2h_balance), detail


def _merge_context_training(engine, train: pd.DataFrame, context_path: Path) -> pd.DataFrame:
    history = load_context_matches(engine, context_path)
    cols = ["match_id", FORM_FEATURE, H2H_FEATURE]
    context_train = history[cols].copy()
    merged = train.merge(
        context_train, on="match_id", how="left", validate="one_to_one", indicator=True
    )
    missing = int(merged["_merge"].ne("both").sum())
    if missing:
        raise ValueError(
            f"{missing} pre-cutoff training matches have no leakage-aware context features. "
            "Rebuild match_context_features.csv before generating this forecast."
        )
    merged = merged.drop(columns=["_merge"])
    if not np.isfinite(merged[[FORM_FEATURE, H2H_FEATURE]].to_numpy(dtype=float)).all():
        raise ValueError("Training context features contain non-finite values")
    return merged


def _top_scorelines(matrix: np.ndarray, count: int = 5) -> list[dict[str, Any]]:
    flat = np.asarray(matrix, dtype=float).ravel()
    indices = np.argsort(flat)[::-1][:count]
    width = matrix.shape[1]
    return [
        {"home_goals": int(index // width), "away_goals": int(index % width),
         "probability": float(flat[index])}
        for index in indices
    ]


def generate_forecast(engine, match_id: int, lineup_type: str,
                      context_path: Path, history_path: Path, player_features_path: Path,
                      player_dataset_path: Path, player_metadata_path: Path,
                      lineup_store: Path) -> dict[str, Any]:
    # Reuse the existing lineup generator to validate the saved XI, calculate
    # the exact lineup z-features, and enforce a pre-kickoff snapshot.
    lineup_source = make_forecast(
        engine, match_id, lineup_type, history_path, player_features_path,
        player_dataset_path, player_metadata_path, lineup_store,
    )
    if not lineup_source.get("player_adjustment_applied"):
        raise ValueError(
            "The source lineup forecast did not apply the player adjustment; "
            "cannot label this as the combined form+H2H+player model."
        )
    for side in ("home", "away"):
        feature = lineup_source.get(f"{side}_lineup_feature", {})
        if feature.get("feature_status") != "lineup_delta_available":
            raise ValueError(
                f"{side.title()} lineup feature is not available "
                f"({feature.get('feature_status')}); refusing to produce a combined player forecast."
            )

    fixture_frame = load_upcoming_fixture(engine, match_id)
    if len(fixture_frame) != 1:
        raise ValueError(f"Expected one upcoming fixture for match_id={match_id}")
    fixture = fixture_frame.iloc[0]
    kickoff = pd.Timestamp(fixture["kickoff_utc"]).tz_convert("UTC")
    target_date = kickoff.floor("D")
    now = pd.Timestamp.now(tz="UTC")
    if now >= kickoff:
        raise ValueError("Fixture has kicked off; refusing to generate a pre-match forecast")

    frozen = load_frozen_player_params(player_metadata_path)
    train = load_finished_training(engine, target_date, player_dataset_path, frozen)
    train = _merge_context_training(engine, train, context_path)
    if len(train) < MIN_TRAIN_MATCHES:
        raise ValueError(f"Only {len(train)} training matches available; need {MIN_TRAIN_MATCHES}")

    form_value, h2h_value, context_detail = _read_target_context(context_path, fixture, target_date)
    model = ContextLineupDixonColes((FORM_FEATURE, H2H_FEATURE), theta=frozen["theta"])
    model.fit(train, reference_date=target_date)

    home_feature = lineup_source["home_lineup_feature"]
    away_feature = lineup_source["away_lineup_feature"]
    x_home = float(home_feature["lineup_delta_standardized"])
    x_away = float(away_feature["lineup_delta_standardized"])
    prediction = model.predict_with_lineup(
        int(fixture["home_team_id"]), int(fixture["away_team_id"]),
        (form_value, h2h_value), x_home, x_away, MAX_GOALS,
    )
    if prediction is None:
        raise ValueError(
            "One or both fixture teams are absent from the training model; "
            "no neutral prediction was substituted."
        )
    probs, lam, mu, matrix = prediction
    probs = tuple(map(float, probs))
    matrix = np.asarray(matrix, dtype=float)
    validate_distribution(matrix, probs)

    record = {
        "schema_version": 1,
        "forecast_model_version": MODEL_VERSION,
        "base_model_version": BASE_MODEL_VERSION,
        "prediction_source": "experimental_dixon_coles_form_h2h_frozen_player_lineup",
        "match_id": int(match_id),
        "season": str(fixture["season"]),
        "matchweek": int(fixture["matchweek"]) if pd.notna(fixture["matchweek"]) else None,
        "kickoff_utc": kickoff.isoformat(),
        "home_team": str(fixture["home_team"]),
        "away_team": str(fixture["away_team"]),
        "lineup_type": lineup_type,
        "lineup_revision": int(lineup_source["lineup_revision"]),
        "lineup_saved_at_utc": lineup_source["lineup_saved_at_utc"],
        "source_player_forecast_hash": lineup_source["forecast_hash"],
        "forecast_generated_at_utc": now.isoformat(),
        "training_matches": int(len(train)),
        "training_cutoff_utc_exclusive": target_date.isoformat(),
        "training_latest_match_utc": pd.to_datetime(train["kickoff_utc"], utc=True).max().isoformat(),
        "context_features": context_detail,
        "lineup_features": {"home": home_feature, "away": away_feature},
        "frozen_player_theta": float(frozen["theta"]),
        "frozen_player_scaler_mean": float(frozen["feature_scaler_mean"]),
        "frozen_player_scaler_std": float(frozen["feature_scaler_std"]),
        "fitted_context_coefficients": {
            FORM_FEATURE: float(model.coef[0]),
            H2H_FEATURE: float(model.coef[1]),
        },
        "expected_goals": {"home": float(lam), "away": float(mu)},
        "probabilities": {"home_win": probs[0], "draw": probs[1], "away_win": probs[2]},
        "scoreline_matrix": matrix.tolist(),
        "top_scorelines": _top_scorelines(matrix),
        "research_note": (
            "Experimental forecast only. Context coefficients are fitted on pre-target-date matches; "
            "the player coefficient and scaler are frozen from the prior player-availability experiment. "
            "This forecast does not alter locked production predictions."
        ),
    }
    hash_payload = {key: value for key, value in record.items()
                    if key not in {"forecast_generated_at_utc", "forecast_hash"}}
    record["forecast_hash"] = canonical_hash(hash_payload)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--lineup-type", choices=["projected", "official_lineup"], required=True)
    parser.add_argument("--context-dataset", default=str(CONTEXT_PATH))
    parser.add_argument("--history", default=str(PLAYER_HISTORY_PATH))
    parser.add_argument("--features", default=str(PLAYER_FEATURES_PATH))
    parser.add_argument("--player-dataset", default=str(PLAYER_MODEL_DATASET_PATH))
    parser.add_argument("--player-metadata", default=str(PLAYER_MODEL_METADATA_PATH))
    parser.add_argument("--lineup-store", default=str(LINEUP_STORE))
    parser.add_argument("--forecast-store", default=str(FORECAST_STORE))
    parser.add_argument("--save", action="store_true", help="append to the separate experimental JSONL store")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = get_sqlalchemy_engine()
    try:
        record = generate_forecast(
            engine, args.match_id, args.lineup_type, Path(args.context_dataset),
            Path(args.history), Path(args.features), Path(args.player_dataset),
            Path(args.player_metadata), Path(args.lineup_store),
        )
        store = Path(args.forecast_store)
        if args.save:
            store.parent.mkdir(parents=True, exist_ok=True)
            existing_hashes = {row.get("forecast_hash") for row in []}
            if store.is_file():
                with store.open("r", encoding="utf-8") as handle:
                    existing_hashes = {
                        item.get("forecast_hash") for line in handle
                        if line.strip() and (item := json.loads(line)).get("forecast_hash")
                    }
            if record["forecast_hash"] in existing_hashes:
                print("This exact combined forecast already exists; no duplicate was appended.")
            else:
                append_jsonl(store, record)
                print(f"Saved separate experimental forecast to {store}")
        else:
            print("DRY RUN: not saved. Add --save to append to the separate experimental JSONL store.")

        print("\nCOMBINED FORM + H2H + LINEUP FORECAST")
        print(f"Fixture: {record['home_team']} vs {record['away_team']} (match_id={record['match_id']})")
        print(f"Kickoff UTC: {record['kickoff_utc']}")
        print(f"Lineup type/revision: {record['lineup_type']} / {record['lineup_revision']}")
        print(f"Training matches: {record['training_matches']}; cutoff exclusive={record['training_cutoff_utc_exclusive']}")
        print(f"Form signal: {record['context_features'][FORM_FEATURE]:+.4f}; coefficient={record['fitted_context_coefficients'][FORM_FEATURE]:+.4f}")
        print(f"H2H signal: {record['context_features'][H2H_FEATURE]:+.4f}; coefficient={record['fitted_context_coefficients'][H2H_FEATURE]:+.4f}")
        print(f"Frozen player theta: {record['frozen_player_theta']:+.6f}")
        print("Probabilities: home={home_win:.1%}, draw={draw:.1%}, away={away_win:.1%}".format(**record["probabilities"]))
        print("Expected goals: home={home:.3f}, away={away:.3f}".format(**record["expected_goals"]))
        print("Top scorelines:")
        for score in record["top_scorelines"][:3]:
            print(f"  {score['home_goals']}-{score['away_goals']}: {score['probability']:.1%}")
        print(f"Forecast hash: {record['forecast_hash']}")
        print("Separate experimental store; locked baseline predictions were not modified.")
        return 0
    except Exception as exc:
        LOG.exception("Combined context-lineup forecast failed: %s", exc)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())