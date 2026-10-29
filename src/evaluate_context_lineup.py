#!/usr/bin/env python3
"""Research-only walk-forward comparison of context and lineup features.

Models evaluated on the same test fixtures with valid historical lineup features:
  A) production Dixon-Coles baseline
  B) baseline + recent-form win-rate difference
  C) baseline + recent form + head-to-head win balance
  D) baseline + frozen lineup adjustment
  E) baseline + recent form + head-to-head + frozen lineup adjustment

The existing production model and PostgreSQL prediction tables are never modified.
Run from repository root:
    python -m src.evaluate_context_lineup --max-dates 2 --bootstrap-reps 0
    python -m src.evaluate_context_lineup
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.db_engine import get_sqlalchemy_engine
from src.evaluate_match_context import (
    CONTEXT_RIDGE,
    FORM_FEATURE,
    H2H_FEATURE,
    FEATURE_NAMES,
    MAX_GOALS,
    MIN_TRAIN_MATCHES,
    RIDGE,
    RHO_BOUNDS,
    TIME_DECAY,
    ContextDixonColes,
    _prediction_with_fallback,
    load_match_data as load_context_matches,
    league_average_rates,
    score_probs,
    validate_distribution,
)
from src.evaluate_player_availability import (
    ResearchDixonColes,
    add_scaled_features,
    load_all_matches as load_player_matches,
)
from src.metrics import outcome_from_goals
from src.models.dixon_coles import DixonColesModel, outcome_probs, score_matrix

LOG = logging.getLogger("evaluate_context_lineup")
EVALUATION_SEASONS = ("2023-2024", "2024-2025")
CONTEXT_NAMES = (FORM_FEATURE, H2H_FEATURE)
MODEL_PREFIXES = (
    ("dixon_coles_v2", "baseline"),
    ("dixon_coles_form_v1", "form"),
    ("dixon_coles_form_h2h_v1", "form_h2h"),
    ("dixon_coles_player_v1", "player"),
    ("dixon_coles_form_h2h_player_v1", "form_h2h_player"),
)
METRICS = ("brier", "rps", "log_loss")
COMPARISONS = (
    ("form", "baseline"),
    ("form_h2h", "baseline"),
    ("form_h2h", "form"),
    ("player", "baseline"),
    ("form_h2h_player", "baseline"),
    ("form_h2h_player", "form_h2h"),
    ("form_h2h_player", "player"),
)


class ContextLineupDixonColes(ContextDixonColes):
    """Context Dixon-Coles model with an externally frozen lineup coefficient.

    Context coefficients are fitted on each date's prior training set. The
    player-lineup slope is loaded from the existing frozen player-model metadata
    and is not re-estimated on the evaluation seasons.
    """

    def __init__(self, feature_names: tuple[str, ...], theta: float,
                 ridge: float = RIDGE, context_ridge: float = CONTEXT_RIDGE):
        super().__init__(feature_names, ridge=ridge, context_ridge=context_ridge)
        self.theta = float(theta)

    def fit(self, train: pd.DataFrame, reference_date: pd.Timestamp):
        needed = {
            "kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals",
            "player_x_home_z", "player_x_away_z", *self.feature_names,
        }
        missing = sorted(needed - set(train.columns))
        if missing:
            raise ValueError(f"Combined model training frame missing columns: {missing}")
        train = train.copy()
        train["kickoff_utc"] = pd.to_datetime(train["kickoff_utc"], utc=True, errors="coerce")
        train = train.dropna(subset=[
            "kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals",
            *self.feature_names,
        ]).copy()
        if len(train) < MIN_TRAIN_MATCHES:
            raise ValueError(f"Need at least {MIN_TRAIN_MATCHES} clean training matches; got {len(train)}")

        team_ids = sorted(
            set(train["home_team_id"].astype(int)) | set(train["away_team_id"].astype(int))
        )
        if len(team_ids) < 2:
            raise ValueError("At least two teams are required")
        self.team_index = {team_id: index for index, team_id in enumerate(team_ids)}
        h_idx = train["home_team_id"].astype(int).map(self.team_index).to_numpy(dtype=int)
        a_idx = train["away_team_id"].astype(int).map(self.team_index).to_numpy(dtype=int)
        home_goals = train["home_goals"].to_numpy(dtype=float)
        away_goals = train["away_goals"].to_numpy(dtype=float)
        kickoff = pd.to_datetime(train["kickoff_utc"], utc=True)
        ref = pd.Timestamp(reference_date)
        ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref.tz_convert("UTC")
        age_days = ((ref - kickoff).dt.total_seconds() / 86400.0).clip(lower=0.0).to_numpy()
        weights = np.exp(-TIME_DECAY * age_days)
        x_context = train[list(self.feature_names)].to_numpy(dtype=float)
        x_home = pd.to_numeric(train["player_x_home_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_away = pd.to_numeric(train["player_x_away_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not np.isfinite(x_context).all() or not np.isfinite(x_home).all() or not np.isfinite(x_away).all():
            raise ValueError("Non-finite context/lineup feature reached combined model fit")

        n_teams = len(team_ids)
        n_features = len(self.feature_names)
        home_mean = max(float(np.average(home_goals, weights=weights)), 0.05)
        away_mean = max(float(np.average(away_goals, weights=weights)), 0.05)
        initial = np.zeros((n_teams - 1) + n_teams + 2 + n_features, dtype=float)
        initial[(n_teams - 1):(n_teams - 1 + n_teams)] = math.log(away_mean)
        gamma_idx = (n_teams - 1) + n_teams
        initial[gamma_idx] = math.log(home_mean / away_mean)
        bounds = (
            [(None, None)] * ((n_teams - 1) + n_teams)
            + [(-1.5, 1.5), RHO_BOUNDS]
            + [(-1.5, 1.5)] * n_features
        )

        def objective(params: np.ndarray):
            attacks, defences, gamma, rho, coefs = self._unpack(params, n_teams, n_features)
            context_delta = x_context @ coefs if n_features else np.zeros(len(train), dtype=float)
            eta_h = attacks[h_idx] + defences[a_idx] + gamma + context_delta + self.theta * x_home
            eta_a = attacks[a_idx] + defences[h_idx] - context_delta + self.theta * x_away
            if (
                not np.isfinite(eta_h).all() or not np.isfinite(eta_a).all()
                or np.max(eta_h) > 15 or np.max(eta_a) > 15
                or np.min(eta_h) < -15 or np.min(eta_a) < -15
            ):
                return 1e100, np.zeros_like(params)
            with np.errstate(over="ignore", invalid="ignore"):
                lam = np.exp(eta_h)
                mu = np.exp(eta_a)
            if not np.isfinite(lam).all() or not np.isfinite(mu).all():
                return 1e100, np.zeros_like(params)

            tau = np.ones(len(train), dtype=float)
            tau_eta_h = np.zeros(len(train), dtype=float)
            tau_eta_a = np.zeros(len(train), dtype=float)
            tau_rho = np.zeros(len(train), dtype=float)
            is00 = (home_goals == 0) & (away_goals == 0)
            is01 = (home_goals == 0) & (away_goals == 1)
            is10 = (home_goals == 1) & (away_goals == 0)
            is11 = (home_goals == 1) & (away_goals == 1)
            tau[is00] = 1.0 - lam[is00] * mu[is00] * rho
            tau_eta_h[is00] = -lam[is00] * mu[is00] * rho / tau[is00]
            tau_eta_a[is00] = tau_eta_h[is00]
            tau_rho[is00] = -lam[is00] * mu[is00] / tau[is00]
            tau[is01] = 1.0 + lam[is01] * rho
            tau_eta_h[is01] = lam[is01] * rho / tau[is01]
            tau_rho[is01] = lam[is01] / tau[is01]
            tau[is10] = 1.0 + mu[is10] * rho
            tau_eta_a[is10] = mu[is10] * rho / tau[is10]
            tau_rho[is10] = mu[is10] / tau[is10]
            tau[is11] = 1.0 - rho
            tau_rho[is11] = -1.0 / tau[is11]
            if not np.isfinite(tau).all() or (tau <= 1e-10).any():
                return 1e100, np.zeros_like(params)

            loglik = (
                home_goals * eta_h - lam - gammaln(home_goals + 1)
                + away_goals * eta_a - mu - gammaln(away_goals + 1)
                + np.log(tau)
            )
            if not np.isfinite(loglik).all():
                return 1e100, np.zeros_like(params)
            g_eta_h = home_goals - lam + tau_eta_h
            g_eta_a = away_goals - mu + tau_eta_a
            weighted_gh = weights * g_eta_h
            weighted_ga = weights * g_eta_a
            attack_grad_full = -(
                np.bincount(h_idx, weights=weighted_gh, minlength=n_teams)
                + np.bincount(a_idx, weights=weighted_ga, minlength=n_teams)
            ) + self.ridge * attacks
            defence_grad = -(
                np.bincount(a_idx, weights=weighted_gh, minlength=n_teams)
                + np.bincount(h_idx, weights=weighted_ga, minlength=n_teams)
            ) + self.ridge * defences
            gamma_grad = -float(np.sum(weighted_gh))
            rho_grad = -float(np.sum(weights * tau_rho))
            grad_parts = [
                attack_grad_full[:-1] - attack_grad_full[-1],
                defence_grad,
                np.array([gamma_grad, rho_grad]),
            ]
            if n_features:
                context_grad = -np.sum((weighted_gh - weighted_ga)[:, None] * x_context, axis=0)
                context_grad += self.context_ridge * coefs
                grad_parts.append(context_grad)
            penalty = 0.5 * self.ridge * (
                float(np.sum(attacks ** 2)) + float(np.sum(defences ** 2))
            )
            penalty += 0.5 * self.context_ridge * float(np.sum(coefs ** 2))
            nll = -float(np.sum(weights * loglik)) + penalty
            grad = np.concatenate(grad_parts)
            if not np.isfinite(nll) or not np.isfinite(grad).all():
                return 1e100, np.zeros_like(params)
            return nll, grad

        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-6, "maxls": 50},
        )
        if not result.success:
            raise RuntimeError(
                f"Combined context-lineup optimisation failed: {result.message}; "
                f"iterations={result.nit}; objective={result.fun}"
            )
        attacks, defences, gamma, rho, coefs = self._unpack(result.x, n_teams, n_features)
        self.attacks = attacks
        self.defences = defences
        self.home_advantage = gamma
        self.rho = rho
        self.coef = np.asarray(coefs, dtype=float)
        self.n_train = len(train)
        self.converged = True
        self.optimization_message = str(result.message)
        return self

    def predict_with_lineup(self, home_team_id: int, away_team_id: int,
                            context_values: Iterable[float], x_home_z: float,
                            x_away_z: float, max_goals: int = MAX_GOALS):
        if int(home_team_id) not in self.team_index or int(away_team_id) not in self.team_index:
            return None
        values = np.asarray(tuple(context_values), dtype=float)
        if values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError("Wrong shape or invalid combined context feature values")
        if not np.isfinite([x_home_z, x_away_z]).all():
            raise ValueError("Invalid lineup features supplied to combined prediction")
        hi = self.team_index[int(home_team_id)]
        ai = self.team_index[int(away_team_id)]
        delta = float(np.dot(self.coef, values))
        lam = math.exp(float(self.attacks[hi] + self.defences[ai] + self.home_advantage + delta + self.theta * x_home_z))
        mu = math.exp(float(self.attacks[ai] + self.defences[hi] - delta + self.theta * x_away_z))
        matrix = score_matrix(lam, mu, self.rho, max_goals)
        probs = tuple(map(float, outcome_probs(matrix)))
        return probs, lam, mu, matrix


def load_frozen_player_metadata(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen lineup model metadata not found: {path}. "
            "Run the existing full player-availability backtest first."
        )
    frame = pd.read_csv(path)
    needed = {"theta", "feature_scaler_mean", "feature_scaler_std"}
    missing = sorted(needed - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Lineup metadata is empty or missing fields: {missing}")
    row = frame.iloc[-1]
    result = {name: float(row[name]) for name in needed}
    if not np.isfinite(list(result.values())).all() or result["feature_scaler_std"] <= 0:
        raise ValueError(f"Invalid frozen lineup metadata values: {result}")
    return result


def load_paired_matches(engine, context_path: Path, player_dataset_path: Path,
                        player_metadata_path: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, int]]:
    context_matches = load_context_matches(engine, context_path)
    player_matches = load_player_matches(engine, player_dataset_path)

    # Verify both pipelines refer to the same database fixture/team orientation.
    identity = context_matches[["match_id", "home_team_id", "away_team_id"]].merge(
        player_matches[["match_id", "home_team_id", "away_team_id"]],
        how="outer", on="match_id", validate="one_to_one", indicator=True,
        suffixes=("_context", "_player"),
    )
    if (identity["_merge"] != "both").any():
        raise ValueError("Context and lineup data do not contain the same completed-match IDs")
    if (
        identity["home_team_id_context"].ne(identity["home_team_id_player"]).any()
        or identity["away_team_id_context"].ne(identity["away_team_id_player"]).any()
    ):
        raise ValueError("Context and lineup datasets disagree on home/away team IDs")

    # Do not use a lineup feature when its source's UTC fixture date disagrees
    # with the database/context cutoff date. The context evaluator itself checks
    # that feature_cutoff_utc_date matches the database target date.
    date_lookup = context_matches.set_index("match_id")["feature_cutoff_utc_date"].astype(str)
    player_matches["context_cutoff_day"] = player_matches["match_id"].map(date_lookup)
    player_day = player_matches["feature_match_date_utc"].dt.strftime("%Y-%m-%d")
    date_aligned = player_day.notna() & player_day.eq(player_matches["context_cutoff_day"])
    player_matches["player_feature_ready"] = (
        player_matches["player_feature_ready"].fillna(False).astype(bool)
        & date_aligned
        & ~player_matches["_feature_date_mismatch"].fillna(False).astype(bool)
    )
    metadata = load_frozen_player_metadata(player_metadata_path)
    player_matches = add_scaled_features(
        player_matches, metadata["feature_scaler_mean"], metadata["feature_scaler_std"]
    )

    selected_player_cols = [
        "match_id", "player_feature_ready", "player_x_home_raw", "player_x_away_raw",
        "player_x_home_z", "player_x_away_z", "_feature_date_mismatch",
    ]
    paired = context_matches.merge(
        player_matches[selected_player_cols], how="left", on="match_id", validate="one_to_one",
    )
    paired["player_feature_ready"] = paired["player_feature_ready"].fillna(False).astype(bool)
    for column in ("player_x_home_raw", "player_x_away_raw", "player_x_home_z", "player_x_away_z"):
        paired[column] = pd.to_numeric(paired[column], errors="coerce").fillna(0.0)
    paired["player_date_mismatch"] = paired["_feature_date_mismatch"].fillna(False).astype(bool)

    counts = {
        "finished_matches": int(len(paired)),
        "lineup_ready_matches_all_seasons": int(paired["player_feature_ready"].sum()),
        "lineup_date_mismatch_rows": int(paired["player_date_mismatch"].sum()),
    }
    return paired.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True), metadata, counts


def predict_player_model(model: ResearchDixonColes, train: pd.DataFrame,
                         reference: pd.Timestamp, match: object,
                         theta: float) -> tuple[tuple[float, float, float], float, float, np.ndarray, bool]:
    home_id, away_id = int(match.home_team_id), int(match.away_team_id)
    x_home = float(match.player_x_home_z)
    x_away = float(match.player_x_away_z)
    predicted = model.predict_probs(home_id, away_id, x_home, x_away, MAX_GOALS)
    if predicted is not None:
        probs, lam, mu, matrix = predicted
        validate_distribution(matrix, probs)
        return probs, float(lam), float(mu), matrix, False
    lam0, mu0 = league_average_rates(train, reference)
    lam, mu = lam0 * math.exp(theta * x_home), mu0 * math.exp(theta * x_away)
    matrix = score_matrix(lam, mu, 0.0, MAX_GOALS)
    probs = tuple(map(float, outcome_probs(matrix)))
    validate_distribution(matrix, probs)
    return probs, lam, mu, matrix, True


def predict_combined_model(model: ContextLineupDixonColes, train: pd.DataFrame,
                           reference: pd.Timestamp, match: object,
                           context_values: tuple[float, ...], theta: float):
    home_id, away_id = int(match.home_team_id), int(match.away_team_id)
    x_home = float(match.player_x_home_z)
    x_away = float(match.player_x_away_z)
    predicted = model.predict_with_lineup(home_id, away_id, context_values, x_home, x_away, MAX_GOALS)
    if predicted is not None:
        probs, lam, mu, matrix = predicted
        validate_distribution(matrix, probs)
        return probs, float(lam), float(mu), matrix, False
    lam0, mu0 = league_average_rates(train, reference)
    delta = float(np.dot(model.coef, np.asarray(context_values, dtype=float)))
    lam = lam0 * math.exp(delta + theta * x_home)
    mu = mu0 * math.exp(-delta + theta * x_away)
    matrix = score_matrix(lam, mu, 0.0, MAX_GOALS)
    probs = tuple(map(float, outcome_probs(matrix)))
    validate_distribution(matrix, probs)
    return probs, lam, mu, matrix, True


def run_experiment(all_matches: pd.DataFrame, seasons: tuple[str, ...], theta: float,
                   max_dates: int | None = None, min_train_matches: int = MIN_TRAIN_MATCHES):
    records: list[dict] = []
    dates_considered = 0
    dates_with_predictions = 0
    skipped_short_history = 0
    eligible_all_test = 0
    lineup_missing_test = 0
    cold_start_count = {prefix: 0 for _, prefix in MODEL_PREFIXES}

    for season in seasons:
        season_rows = all_matches[all_matches["season_key"].eq(season)].copy()
        if season_rows.empty:
            raise ValueError(f"No finished fixtures found for held-out season {season}")
        all_dates = sorted(season_rows["_match_date"].unique())
        if max_dates is not None:
            all_dates = all_dates[:max_dates]
        dates_considered += len(all_dates)

        for eval_date in all_dates:
            train = all_matches[all_matches["_match_date"] < eval_date].copy()
            test_day_all = season_rows[season_rows["_match_date"].eq(eval_date)].copy()
            eligible_all_test += len(test_day_all)
            test_day = test_day_all[test_day_all["player_feature_ready"]].copy()
            lineup_missing_test += len(test_day_all) - len(test_day)
            if len(train) < min_train_matches:
                skipped_short_history += len(test_day)
                LOG.warning("Skipping %s %s: only %d training matches", season, eval_date, len(train))
                continue
            if test_day.empty:
                continue
            reference = pd.Timestamp(eval_date).tz_localize("UTC")
            dates_with_predictions += 1

            baseline = DixonColesModel(time_decay=TIME_DECAY, ridge=RIDGE)
            baseline.fit(train, reference_date=reference)
            form_model = ContextDixonColes((FORM_FEATURE,)).fit(train, reference)
            context_model = ContextDixonColes(CONTEXT_NAMES).fit(train, reference)
            player_model = ResearchDixonColes().fit(train, reference_date=reference, fixed_theta=theta)
            combined_model = ContextLineupDixonColes(CONTEXT_NAMES, theta=theta).fit(train, reference)

            LOG.info(
                "Evaluating %s %s: train=%d paired_test=%d form_beta=%.4f context_betas=%s lineup_theta=%.4f",
                season, eval_date, len(train), len(test_day), float(form_model.coef[0]),
                np.round(context_model.coef, 4).tolist(), theta,
            )

            for match in test_day.itertuples(index=False):
                outcome = outcome_from_goals(match.home_goals, match.away_goals)
                x_form = float(getattr(match, FORM_FEATURE))
                x_h2h = float(getattr(match, H2H_FEATURE))
                context_form = (x_form,)
                context_both = (x_form, x_h2h)

                predictions = {}
                predictions["baseline"] = _prediction_with_fallback(
                    None, baseline, train, reference, match, tuple(), "baseline"
                )
                predictions["form"] = _prediction_with_fallback(
                    form_model, baseline, train, reference, match, context_form, "form"
                )
                predictions["form_h2h"] = _prediction_with_fallback(
                    context_model, baseline, train, reference, match, context_both, "form_h2h"
                )
                predictions["player"] = predict_player_model(
                    player_model, train, reference, match, theta
                )
                predictions["form_h2h_player"] = predict_combined_model(
                    combined_model, train, reference, match, context_both, theta
                )

                rec = {}
                for prefix, pred in predictions.items():
                    probs, lam, mu, matrix, fallback = pred
                    cold_start_count[prefix] += int(fallback)
                    scores = score_probs(probs, outcome)
                    # For clear downstream auditability, persist probabilities,
                    # expected goals and scores separately for each model variant.
                    if "rec" not in locals():
                        rec = {}
                    rec.update({
                        f"{prefix}_p_home": probs[0],
                        f"{prefix}_p_draw": probs[1],
                        f"{prefix}_p_away": probs[2],
                        f"{prefix}_expected_home_goals": lam,
                        f"{prefix}_expected_away_goals": mu,
                        f"{prefix}_brier": scores["brier"],
                        f"{prefix}_rps": scores["rps"],
                        f"{prefix}_log_loss": scores["log_loss"],
                        f"{prefix}_cold_start_fallback": bool(fallback),
                    })
                records.append({
                    "match_id": int(match.match_id),
                    "season": season,
                    "match_date_utc": str(eval_date),
                    "matchweek": getattr(match, "matchweek", None),
                    "home_name": match.home_name,
                    "away_name": match.away_name,
                    "home_goals": int(match.home_goals),
                    "away_goals": int(match.away_goals),
                    "actual_outcome": outcome,
                    "n_train": int(len(train)),
                    "train_cutoff_date_exclusive": str(eval_date),
                    FORM_FEATURE: x_form,
                    H2H_FEATURE: x_h2h,
                    "player_x_home_z": float(match.player_x_home_z),
                    "player_x_away_z": float(match.player_x_away_z),
                    "player_theta_frozen": float(theta),
                    "form_beta": float(form_model.coef[0]),
                    "combined_form_beta": float(context_model.coef[0]),
                    "combined_h2h_beta": float(context_model.coef[1]),
                    "combined_player_form_beta": float(combined_model.coef[0]),
                    "combined_player_h2h_beta": float(combined_model.coef[1]),
                    **rec,
                })

    results = pd.DataFrame.from_records(records)
    if results.empty:
        raise RuntimeError(
            "No lineup-ready fixtures were evaluated. Check player dataset coverage, "
            "the feature-date alignment, and held-out seasons."
        )
    metadata = {
        "evaluation_seasons": ",".join(seasons),
        "matches_evaluated_paired_lineup_ready": int(len(results)),
        "held_out_finished_fixtures_considered": int(eligible_all_test),
        "held_out_fixtures_excluded_no_valid_lineup_features": int(lineup_missing_test),
        "evaluation_dates_considered": int(dates_considered),
        "dates_with_paired_predictions": int(dates_with_predictions),
        "skipped_for_insufficient_training": int(skipped_short_history),
        "frozen_player_theta": float(theta),
        "min_train_matches": int(min_train_matches),
        "max_dates_per_season": max_dates if max_dates is not None else "all",
        "history_cutoff_rule": "all matches on target UTC calendar date excluded from training; lineup feature date must match target date",
        "comparison_sample": "all five models use the exact same held-out fixtures with date-aligned valid historical lineup features",
    }
    metadata.update({f"cold_start_fallbacks_{key}": int(value) for key, value in cold_start_count.items()})
    return results, metadata


def create_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = [(season, results[results["season"].eq(season)]) for season in sorted(results["season"].unique())]
    scopes.append(("ALL_HELD_OUT", results))
    rows = []
    for scope, frame in scopes:
        for model, prefix in MODEL_PREFIXES:
            rows.append({
                "scope": scope, "model": model, "n": len(frame),
                **{metric: float(frame[f"{prefix}_{metric}"].mean()) for metric in METRICS},
            })
    diff_rows = []
    for scope, frame in scopes:
        for challenger, control in COMPARISONS:
            for metric in METRICS:
                delta = frame[f"{challenger}_{metric}"] - frame[f"{control}_{metric}"]
                diff_rows.append({
                    "scope": scope,
                    "comparison": f"{challenger}_minus_{control}",
                    "metric": metric,
                    "n": int(delta.notna().sum()),
                    "mean_delta": float(delta.mean()),
                    "interpretation": "negative favours challenger; lower is better",
                })
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def bootstrap_date_clusters(results: pd.DataFrame, reps: int, seed: int = 20261026) -> pd.DataFrame:
    if reps <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    groups = {str(day): frame for day, frame in results.groupby("match_date_utc", sort=True)}
    dates = np.asarray(list(groups.keys()), dtype=object)
    if len(dates) < 2:
        return pd.DataFrame()
    rows = []
    for challenger, control in COMPARISONS:
        for metric in METRICS:
            samples = np.empty(reps, dtype=float)
            for i in range(reps):
                picked = rng.choice(dates, size=len(dates), replace=True)
                sample_vals = []
                for day in picked:
                    frame = groups[str(day)]
                    sample_vals.append((frame[f"{challenger}_{metric}"] - frame[f"{control}_{metric}"]).to_numpy(dtype=float))
                samples[i] = float(np.concatenate(sample_vals).mean())
            observed = float((results[f"{challenger}_{metric}"] - results[f"{control}_{metric}"]).mean())
            rows.append({
                "comparison": f"{challenger}_minus_{control}", "metric": metric,
                "n_matches": int(len(results)), "n_dates": int(len(dates)), "mean_delta": observed,
                "ci_95_low": float(np.quantile(samples, 0.025)),
                "ci_95_high": float(np.quantile(samples, 0.975)),
                "bootstrap_reps": int(reps),
                "interpretation": "negative favours challenger; interval crossing zero means uncertainty remains",
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dataset", default="data/processed/match_context_features.csv",
                        help="Output from src.build_match_context_features")
    parser.add_argument("--player-dataset", default="data/processed/player_model_dataset.csv",
                        help="Output from src.prepare_player_model_dataset")
    parser.add_argument("--player-metadata", default="reports/player_availability_backtest_metadata.csv",
                        help="Existing frozen player coefficient and scaler metadata")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Smoke-test limit applied per held-out season")
    parser.add_argument("--min-train-matches", type=int, default=MIN_TRAIN_MATCHES)
    parser.add_argument("--bootstrap-reps", type=int, default=500,
                        help="Date-cluster bootstrap repetitions; use 0 to skip")
    parser.add_argument("--output", default="reports/context_lineup_backtest.csv",
                        help="Per-fixture paired predictions output CSV")
    parser.add_argument("--seasons", nargs="+", default=list(EVALUATION_SEASONS),
                        help="Held-out season keys, e.g. 2023-2024 2024-2025")
    args = parser.parse_args(argv)
    if args.max_dates is not None and args.max_dates < 1:
        parser.error("--max-dates must be >= 1")
    if args.min_train_matches < MIN_TRAIN_MATCHES:
        parser.error(f"--min-train-matches must be >= {MIN_TRAIN_MATCHES}")
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be >= 0")
    seasons = tuple(str(s) for s in args.seasons)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = get_sqlalchemy_engine()
    try:
        all_matches, metadata, counts = load_paired_matches(
            engine, Path(args.context_dataset), Path(args.player_dataset), Path(args.player_metadata)
        )
    except Exception:
        LOG.exception("Unable to load matched historical results, context features and lineup features")
        return 1
    finally:
        engine.dispose()

    try:
        results, run_metadata = run_experiment(
            all_matches, seasons, metadata["theta"], max_dates=args.max_dates,
            min_train_matches=args.min_train_matches,
        )
        summary, differences = create_summary(results)
        uncertainty = bootstrap_date_clusters(results, args.bootstrap_reps)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_path = output.with_name(output.stem + "_summary.csv")
        diff_path = output.with_name(output.stem + "_differences.csv")
        ci_path = output.with_name(output.stem + "_uncertainty.csv")
        metadata_path = output.with_name(output.stem + "_metadata.csv")
        results.to_csv(output, index=False, float_format="%.10g")
        summary.to_csv(summary_path, index=False, float_format="%.10g")
        differences.to_csv(diff_path, index=False, float_format="%.10g")
        uncertainty.to_csv(ci_path, index=False, float_format="%.10g")
        final_metadata = run_metadata | counts | {
            "player_metadata_path": str(Path(args.player_metadata)),
            "player_dataset_path": str(Path(args.player_dataset)),
            "context_dataset_path": str(Path(args.context_dataset)),
            "feature_scaler_mean": metadata["feature_scaler_mean"],
            "feature_scaler_std": metadata["feature_scaler_std"],
        }
        pd.DataFrame([final_metadata]).to_csv(metadata_path, index=False, float_format="%.10g")

        print("\nCONTEXT + LINEUP EXPANDING-WINDOW EVALUATION")
        print(f"Held-out seasons: {', '.join(seasons)}")
        print(f"All completed fixtures in joined data: {counts['finished_matches']:,}")
        print(f"Lineup-ready fixtures across all joined seasons: {counts['lineup_ready_matches_all_seasons']:,}")
        print(f"Lineup feature source-date mismatches: {counts['lineup_date_mismatch_rows']:,}")
        print(f"Held-out fixtures considered: {run_metadata['held_out_finished_fixtures_considered']:,}")
        print(f"Held-out fixtures excluded without date-aligned lineup features: {run_metadata['held_out_fixtures_excluded_no_valid_lineup_features']:,}")
        print(f"Paired held-out fixtures evaluated across all five models: {run_metadata['matches_evaluated_paired_lineup_ready']:,}")
        print(f"Frozen player coefficient theta: {metadata['theta']:.6f}")
        print("\nMean scores (lower is better):")
        print(summary.round(6).to_string(index=False))
        print("\nPaired score differences (negative favours challenger; lower is better):")
        print(differences[differences["scope"].eq("ALL_HELD_OUT")].round(6).to_string(index=False))
        if not uncertainty.empty:
            print("\nDate-cluster bootstrap 95% intervals for pooled paired score differences:")
            print(uncertainty.round(6).to_string(index=False))
        print("\nReports saved:")
        print(f"  Per-fixture predictions: {output}")
        print(f"  Summary:                 {summary_path}")
        print(f"  Differences:             {diff_path}")
        print(f"  Uncertainty:             {ci_path}")
        print(f"  Metadata:                {metadata_path}")
        print("Research-only: no database writes and no changes to production predictions.")
        return 0
    except Exception:
        LOG.exception("Context + lineup experiment failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
