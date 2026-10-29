#!/usr/bin/env python3
"""Leakage-aware walk-forward test of recent-form and head-to-head features.

This research-only script compares:
  A) the production DixonColesModel baseline,
  B) a Dixon-Coles model with a learned recent-form win-rate feature, and
  C) a Dixon-Coles model with recent form plus head-to-head balance.

It reads finished results from PostgreSQL and match-context features from CSV.
It does not write to PostgreSQL or change production predictions.

Run from the repository root:
    python -m src.evaluate_match_context --max-dates 2 --bootstrap-reps 0
    python -m src.evaluate_match_context

The default held-out seasons (2023-24 and 2024-25) align with the current
player-availability experiment. All models are scored on exactly the same
held-out fixtures. Training excludes all matches on the target UTC calendar day.
Context coefficients and feature transforms are fitted using prior matches only.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sqlalchemy import text

from src.db_engine import get_sqlalchemy_engine
from src.metrics import (
    calculate_brier_score,
    calculate_log_loss,
    calculate_rps,
    outcome_from_goals,
)
from src.models.dixon_coles import DixonColesModel, outcome_probs, score_matrix

LOG = logging.getLogger("evaluate_match_context")

EVALUATION_SEASONS = ("2023-2024", "2024-2025")
TIME_DECAY = 0.001
RIDGE = 1.0
CONTEXT_RIDGE = 2.0
MAX_GOALS = 9
MIN_TRAIN_MATCHES = 200
CONTEXT_BOUNDS = (-1.5, 1.5)
RHO_BOUNDS = (-0.3, 0.3)
FORM_FEATURE = "form_win_rate_diff"
H2H_FEATURE = "h2h_win_balance"
FEATURE_NAMES = (FORM_FEATURE, H2H_FEATURE)

MATCHES_SQL = text("""
    SELECT m.id AS match_id,
           m.season,
           m.matchweek,
           m.kickoff_utc AS kickoff_utc,
           m.home_team_id,
           m.away_team_id,
           ht.name AS home_name,
           awt.name AS away_name,
           m.home_goals,
           m.away_goals
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams awt ON awt.id = m.away_team_id
    WHERE m.status = 'FINISHED'
      AND m.home_goals IS NOT NULL
      AND m.away_goals IS NOT NULL
      AND m.kickoff_utc IS NOT NULL
    ORDER BY m.kickoff_utc ASC, m.id ASC
""")

REQUIRED_CONTEXT_COLUMNS = {
    "match_id", "feature_cutoff_utc_date", "home_team_id", "away_team_id",
    "home_form_matches", "home_form_wins", "away_form_matches", "away_form_wins",
    "h2h_matches", "h2h_home_wins", "h2h_away_wins", "history_rule",
}


def normalize_season(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    match = re.fullmatch(r"(\d{4})\s*[-/]\s*(\d{2}|\d{4})", raw)
    if not match:
        return raw
    start = int(match.group(1))
    tail = match.group(2)
    end = int(tail) if len(tail) == 4 else (start // 100) * 100 + int(tail)
    if end != start + 1:
        return None
    return f"{start}-{end:04d}"


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def _win_rate(wins: float, matches: float, neutral: float = 1.0 / 3.0) -> float:
    if not np.isfinite(matches) or matches <= 0 or not np.isfinite(wins):
        return neutral
    return float(wins / matches)


def load_match_data(engine, context_path: Path) -> pd.DataFrame:
    """Load finished DB matches and join their precomputed context features."""
    if not context_path.is_file():
        raise FileNotFoundError(f"Context feature CSV not found: {context_path}")

    context = pd.read_csv(context_path, low_memory=False)
    missing = sorted(REQUIRED_CONTEXT_COLUMNS - set(context.columns))
    if missing:
        raise ValueError(f"Context CSV is missing required columns: {missing}")
    context["match_id"] = pd.to_numeric(context["match_id"], errors="coerce")
    context = context.dropna(subset=["match_id"]).copy()
    context["match_id"] = context["match_id"].astype(int)
    if context["match_id"].duplicated().any():
        examples = context.loc[context["match_id"].duplicated(keep=False), "match_id"].head(10).tolist()
        raise ValueError(f"Context CSV contains duplicate match IDs, e.g. {examples}")

    # All features used below are results-only features created before the target
    # UTC date; there must be no player XIs or target-match outcomes in them.
    for column in REQUIRED_CONTEXT_COLUMNS - {"match_id", "feature_cutoff_utc_date", "history_rule"}:
        context[column] = pd.to_numeric(context[column], errors="coerce")
    context["feature_cutoff_utc_date"] = context["feature_cutoff_utc_date"].astype(str)
    rules = set(context["history_rule"].dropna().astype(str).unique())
    if rules and rules != {"finished_matches_before_target_utc_date_only"}:
        raise ValueError(f"Unexpected context history rule(s): {sorted(rules)}")

    with engine.connect() as connection:
        matches = pd.read_sql_query(MATCHES_SQL, connection)
    if matches.empty:
        raise RuntimeError("No finished, scored matches found in PostgreSQL")
    matches["match_id"] = pd.to_numeric(matches["match_id"], errors="coerce")
    matches = matches.dropna(subset=["match_id"]).copy()
    matches["match_id"] = matches["match_id"].astype(int)
    if matches["match_id"].duplicated().any():
        raise ValueError("Database returned duplicate match IDs")

    matches["kickoff_utc"] = pd.to_datetime(matches["kickoff_utc"], utc=True, errors="coerce")
    matches["season_key"] = matches["season"].map(normalize_season)
    matches["home_team_id"] = pd.to_numeric(matches["home_team_id"], errors="coerce")
    matches["away_team_id"] = pd.to_numeric(matches["away_team_id"], errors="coerce")
    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")
    matches = matches.dropna(subset=[
        "kickoff_utc", "season_key", "home_team_id", "away_team_id", "home_goals", "away_goals"
    ]).copy()
    matches["home_team_id"] = matches["home_team_id"].astype(int)
    matches["away_team_id"] = matches["away_team_id"].astype(int)
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["_match_date"] = matches["kickoff_utc"].dt.strftime("%Y-%m-%d")

    merge_columns = [
        "match_id", "feature_cutoff_utc_date", "home_team_id", "away_team_id",
        "home_form_matches", "home_form_wins", "away_form_matches", "away_form_wins",
        "h2h_matches", "h2h_home_wins", "h2h_away_wins", "history_rule",
    ]
    context_to_merge = context[merge_columns].copy()
    matches = matches.merge(
        context_to_merge,
        how="left",
        on="match_id",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_context"),
    )
    missing_context = matches["_merge"] != "both"
    if missing_context.any():
        sample = matches.loc[missing_context, "match_id"].head(10).tolist()
        raise ValueError(
            f"{int(missing_context.sum())} finished matches have no context row; examples: {sample}. "
            "Rebuild match_context_features.csv first."
        )
    matches = matches.drop(columns=["_merge"])

    # Ensure the context row is for this exact home/away fixture and target UTC date.
    home_mismatch = matches["home_team_id"] != matches["home_team_id_context"] if "home_team_id_context" in matches else pd.Series(False, index=matches.index)
    away_mismatch = matches["away_team_id"] != matches["away_team_id_context"] if "away_team_id_context" in matches else pd.Series(False, index=matches.index)
    if home_mismatch.any() or away_mismatch.any():
        raise ValueError("Context feature team IDs do not match the database fixture orientation")

    # Pandas adds suffixes only when merge fields collide. Here team IDs collide,
    # so the merged context team IDs carry the _context suffix.
    if "home_team_id_context" not in matches or "away_team_id_context" not in matches:
        raise ValueError("Unable to verify context team IDs against the database fixture")
    target_day = matches["kickoff_utc"].dt.strftime("%Y-%m-%d")
    cutoff_day = matches["feature_cutoff_utc_date"].astype(str)
    date_mismatch = target_day.ne(cutoff_day)
    if date_mismatch.any():
        sample = matches.loc[date_mismatch, ["match_id", "kickoff_utc", "feature_cutoff_utc_date"]].head(5)
        raise ValueError(
            "Context cutoff date differs from fixture UTC date. Rebuild the context features. Examples:\n"
            + sample.to_string(index=False)
        )

    # Validate counts before creating numeric signal features.
    count_columns = [
        "home_form_matches", "home_form_wins", "away_form_matches", "away_form_wins",
        "h2h_matches", "h2h_home_wins", "h2h_away_wins",
    ]
    for column in count_columns:
        if matches[column].isna().any() or (matches[column] < 0).any():
            raise ValueError(f"Context column {column} contains missing or negative counts")
    if (matches["home_form_matches"] > 5).any() or (matches["away_form_matches"] > 5).any():
        raise ValueError("Recent-form window contains more than five matches")
    if (matches["home_form_wins"] > matches["home_form_matches"]).any():
        raise ValueError("Home form wins exceed available form matches")
    if (matches["away_form_wins"] > matches["away_form_matches"]).any():
        raise ValueError("Away form wins exceed available form matches")
    if (matches["h2h_matches"] > 5).any():
        raise ValueError("Head-to-head window contains more than five matches")
    if (matches["h2h_home_wins"] + matches["h2h_away_wins"] > matches["h2h_matches"]).any():
        raise ValueError("Head-to-head home/away wins exceed available meetings")

    # Requested form signal: difference between each team's win rate over its
    # last five completed matches. With no history, use a neutral 1/3 win rate.
    home_rate = [
        _win_rate(w, n) for w, n in zip(matches["home_form_wins"], matches["home_form_matches"])
    ]
    away_rate = [
        _win_rate(w, n) for w, n in zip(matches["away_form_wins"], matches["away_form_matches"])
    ]
    matches[FORM_FEATURE] = np.asarray(home_rate, dtype=float) - np.asarray(away_rate, dtype=float)

    # Head-to-head signal is win balance from the target home team's perspective.
    # No prior meetings = neutral signal (0), rather than a fabricated result.
    h2h_n = matches["h2h_matches"].to_numpy(dtype=float)
    h2h_wins_delta = (
        matches["h2h_home_wins"].to_numpy(dtype=float)
        - matches["h2h_away_wins"].to_numpy(dtype=float)
    )
    matches[H2H_FEATURE] = np.divide(
        h2h_wins_delta,
        h2h_n,
        out=np.zeros(len(matches), dtype=float),
        where=h2h_n > 0,
    )

    if not np.isfinite(matches[[FORM_FEATURE, H2H_FEATURE]].to_numpy(dtype=float)).all():
        raise ValueError("Context features contain non-finite values")
    return matches.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def league_average_rates(train: pd.DataFrame, reference_date: pd.Timestamp) -> tuple[float, float]:
    kickoff = pd.to_datetime(train["kickoff_utc"], utc=True)
    age_days = ((reference_date - kickoff).dt.total_seconds() / 86400.0).clip(lower=0).to_numpy()
    weights = np.exp(-TIME_DECAY * age_days)
    home_rate = float(np.average(train["home_goals"].to_numpy(dtype=float), weights=weights))
    away_rate = float(np.average(train["away_goals"].to_numpy(dtype=float), weights=weights))
    if not np.isfinite([home_rate, away_rate]).all() or home_rate <= 0 or away_rate <= 0:
        raise ValueError(f"Invalid league average rates: home={home_rate}, away={away_rate}")
    return home_rate, away_rate


def validate_distribution(matrix: np.ndarray, probs: Iterable[float]) -> None:
    matrix = np.asarray(matrix, dtype=float)
    probs = np.asarray(tuple(probs), dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(probs).all():
        raise ValueError("Model returned non-finite probabilities")
    if (matrix < 0).any() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Model returned probabilities outside valid ranges")
    if abs(float(matrix.sum()) - 1.0) > 1e-8 or abs(float(probs.sum()) - 1.0) > 1e-8:
        raise ValueError("Model probabilities do not sum to one")
    derived = np.asarray(outcome_probs(matrix), dtype=float)
    if np.max(np.abs(derived - probs)) > 1e-8:
        raise ValueError("1X2 probabilities disagree with score matrix")


def score_probs(probs: Iterable[float], outcome: str) -> dict[str, float]:
    home, draw, away = map(float, probs)
    result = {
        "brier": float(calculate_brier_score(home, draw, away, outcome)),
        "rps": float(calculate_rps(home, draw, away, outcome)),
        "log_loss": float(calculate_log_loss(home, draw, away, outcome)),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError(f"Non-finite scoring metrics for {probs} and {outcome}")
    return result


class ContextDixonColes:
    """Research Dixon-Coles model with optional relative form/H2H predictors.

    A single context signal shifts the home and away log scoring rates in
    opposite directions. The model estimates a regularised shared coefficient
    for each enabled signal while refitting team attack/defence and rho.
    """

    def __init__(self, feature_names: tuple[str, ...], ridge: float = RIDGE,
                 context_ridge: float = CONTEXT_RIDGE):
        self.feature_names = tuple(feature_names)
        self.ridge = float(ridge)
        self.context_ridge = float(context_ridge)
        self.team_index: dict[int, int] = {}
        self.attacks = np.array([], dtype=float)
        self.defences = np.array([], dtype=float)
        self.home_advantage = 0.0
        self.rho = 0.0
        self.coef = np.zeros(len(self.feature_names), dtype=float)
        self.n_train = 0
        self.converged = False

    @staticmethod
    def _unpack(params: np.ndarray, n_teams: int, n_features: int):
        n_attack_free = n_teams - 1
        attack_free = params[:n_attack_free]
        attacks = np.concatenate([attack_free, [-float(np.sum(attack_free))]])
        def_start = n_attack_free
        defences = params[def_start:def_start + n_teams]
        gamma_idx = def_start + n_teams
        gamma = float(params[gamma_idx])
        rho = float(params[gamma_idx + 1])
        coefs = params[gamma_idx + 2:gamma_idx + 2 + n_features]
        return attacks, defences, gamma, rho, coefs

    def fit(self, train: pd.DataFrame, reference_date: pd.Timestamp) -> "ContextDixonColes":
        needed = {
            "kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals",
            *self.feature_names,
        }
        missing = sorted(needed - set(train.columns))
        if missing:
            raise ValueError(f"Training frame missing columns: {missing}")
        train = train.copy()
        train["kickoff_utc"] = pd.to_datetime(train["kickoff_utc"], utc=True, errors="coerce")
        train = train.dropna(subset=[
            "kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals", *self.feature_names
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
        x = train[list(self.feature_names)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite context features reached model fit")
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
            + [CONTEXT_BOUNDS] * n_features
        )

        def objective(params: np.ndarray):
            attacks, defences, gamma, rho, coefs = self._unpack(params, n_teams, n_features)
            delta = x @ coefs if n_features else np.zeros(len(train), dtype=float)
            eta_h = attacks[h_idx] + defences[a_idx] + gamma + delta
            eta_a = attacks[a_idx] + defences[h_idx] - delta
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
                context_grad = -np.sum((weighted_gh - weighted_ga)[:, None] * x, axis=0)
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
                f"Context Dixon-Coles optimisation failed: {result.message}; "
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

    def predict_probs(self, home_team_id: int, away_team_id: int,
                      feature_values: Iterable[float], max_goals: int = MAX_GOALS):
        if int(home_team_id) not in self.team_index or int(away_team_id) not in self.team_index:
            return None
        values = np.asarray(tuple(feature_values), dtype=float)
        if values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError("Wrong shape or invalid prediction context feature values")
        hi = self.team_index[int(home_team_id)]
        ai = self.team_index[int(away_team_id)]
        delta = float(np.dot(self.coef, values))
        lam = math.exp(float(self.attacks[hi] + self.defences[ai] + self.home_advantage + delta))
        mu = math.exp(float(self.attacks[ai] + self.defences[hi] - delta))
        matrix = score_matrix(lam, mu, self.rho, max_goals)
        probs = tuple(map(float, outcome_probs(matrix)))
        return probs, lam, mu, matrix


def _prediction_with_fallback(
    model: ContextDixonColes | None,
    base_model: DixonColesModel,
    train: pd.DataFrame,
    reference: pd.Timestamp,
    match: object,
    features: tuple[float, ...],
    variant: str,
):
    """Use a paired league-average fallback for teams absent from training."""
    home_id = int(match.home_team_id)
    away_id = int(match.away_team_id)
    if model is None:
        if home_id in base_model.team_indices and away_id in base_model.team_indices:
            ph, pd_, pa, lam, mu, matrix = base_model.predict_probs(home_id, away_id, MAX_GOALS)
            probs = (float(ph), float(pd_), float(pa))
            validate_distribution(matrix, probs)
            return probs, float(lam), float(mu), matrix, False
        lam, mu = league_average_rates(train, reference)
        matrix = score_matrix(lam, mu, 0.0, MAX_GOALS)
        probs = tuple(map(float, outcome_probs(matrix)))
        validate_distribution(matrix, probs)
        return probs, lam, mu, matrix, True

    predicted = model.predict_probs(home_id, away_id, features, MAX_GOALS)
    if predicted is not None:
        probs, lam, mu, matrix = predicted
        validate_distribution(matrix, probs)
        return probs, float(lam), float(mu), matrix, False

    # Apply the context signal to common league-average scoring rates on cold start.
    # This preserves a paired sample while still letting the experimental feature
    # contribute; the baseline's cold-start prediction stays a neutral league-rate forecast.
    lam0, mu0 = league_average_rates(train, reference)
    delta = float(np.dot(model.coef, np.asarray(features, dtype=float)))
    lam = lam0 * math.exp(delta)
    mu = mu0 * math.exp(-delta)
    matrix = score_matrix(lam, mu, 0.0, MAX_GOALS)
    probs = tuple(map(float, outcome_probs(matrix)))
    validate_distribution(matrix, probs)
    return probs, lam, mu, matrix, True


def run_experiment(all_matches: pd.DataFrame, seasons: tuple[str, ...],
                   max_dates: int | None = None,
                   min_train_matches: int = MIN_TRAIN_MATCHES):
    records: list[dict] = []
    dates_considered = 0
    skipped_short_history = 0
    cold_start_fallbacks = 0

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
            test_day = season_rows[season_rows["_match_date"].eq(eval_date)].copy()
            if len(train) < min_train_matches:
                skipped_short_history += len(test_day)
                LOG.warning("Skipping %s %s: only %d training matches", season, eval_date, len(train))
                continue
            reference = pd.Timestamp(eval_date).tz_localize("UTC")

            # Fit production baseline and both context variants on identical prior data.
            baseline = DixonColesModel(time_decay=TIME_DECAY, ridge=RIDGE)
            baseline.fit(train, reference_date=reference)
            form_model = ContextDixonColes((FORM_FEATURE,)).fit(train, reference)
            combined_model = ContextDixonColes(FEATURE_NAMES).fit(train, reference)
            LOG.info(
                "Evaluating %s %s: train=%d test=%d form_beta=%.4f combined_betas=%s",
                season, eval_date, len(train), len(test_day),
                float(form_model.coef[0]), np.round(combined_model.coef, 4).tolist(),
            )

            for match in test_day.itertuples(index=False):
                outcome = outcome_from_goals(match.home_goals, match.away_goals)
                x_form = float(getattr(match, FORM_FEATURE))
                x_h2h = float(getattr(match, H2H_FEATURE))
                feature_form = (x_form,)
                feature_both = (x_form, x_h2h)
                baseline_pred = _prediction_with_fallback(
                    None, baseline, train, reference, match, tuple(), "baseline"
                )
                form_pred = _prediction_with_fallback(
                    form_model, baseline, train, reference, match, feature_form, "form"
                )
                both_pred = _prediction_with_fallback(
                    combined_model, baseline, train, reference, match, feature_both, "form_h2h"
                )
                bp, blam, bmu, _, b_fallback = baseline_pred
                fp, flam, fmu, _, f_fallback = form_pred
                cp, clam, cmu, _, c_fallback = both_pred
                cold_start_fallbacks += int(b_fallback or f_fallback or c_fallback)
                b_scores = score_probs(bp, outcome)
                f_scores = score_probs(fp, outcome)
                c_scores = score_probs(cp, outcome)

                rec = {
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
                    "form_model_beta": float(form_model.coef[0]),
                    "combined_form_beta": float(combined_model.coef[0]),
                    "combined_h2h_beta": float(combined_model.coef[1]),
                    "baseline_cold_start_fallback": bool(b_fallback),
                    "form_cold_start_fallback": bool(f_fallback),
                    "combined_cold_start_fallback": bool(c_fallback),
                }
                for prefix, probs, lam, mu, scores in (
                    ("baseline", bp, blam, bmu, b_scores),
                    ("form", fp, flam, fmu, f_scores),
                    ("form_h2h", cp, clam, cmu, c_scores),
                ):
                    rec.update({
                        f"{prefix}_p_home": probs[0],
                        f"{prefix}_p_draw": probs[1],
                        f"{prefix}_p_away": probs[2],
                        f"{prefix}_expected_home_goals": lam,
                        f"{prefix}_expected_away_goals": mu,
                        **{f"{prefix}_{key}": value for key, value in scores.items()},
                    })
                records.append(rec)

    results = pd.DataFrame.from_records(records)
    if results.empty:
        raise RuntimeError("No fixtures were evaluated. Check seasons and minimum training size.")
    metadata = {
        "evaluation_seasons": ",".join(seasons),
        "matches_evaluated": int(len(results)),
        "evaluation_dates_considered": int(dates_considered),
        "dates_with_predictions": int(results["match_date_utc"].nunique()),
        "skipped_for_insufficient_training": int(skipped_short_history),
        "cold_start_fallback_predictions": int(cold_start_fallbacks),
        "min_train_matches": int(min_train_matches),
        "max_dates_per_season": max_dates if max_dates is not None else "all",
        "time_decay": TIME_DECAY,
        "team_strength_ridge": RIDGE,
        "context_coefficient_ridge": CONTEXT_RIDGE,
        "history_cutoff_rule": "all matches on target UTC calendar date excluded from training",
        "form_feature": "home win rate minus away win rate over each team's last five completed matches; no history uses neutral 1/3 rate",
        "h2h_feature": "(target-home wins minus target-away wins) / previous meetings; no meetings uses 0",
    }
    return results, metadata


def create_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [
        ("dixon_coles_v2", "baseline"),
        ("dixon_coles_form_v1", "form"),
        ("dixon_coles_form_h2h_v1", "form_h2h"),
    ]
    metrics = ("brier", "rps", "log_loss")
    scopes: list[tuple[str, pd.DataFrame]] = [
        (season, results[results["season"] == season])
        for season in sorted(results["season"].unique())
    ] + [("ALL_HELD_OUT", results)]
    summary_rows = []
    for scope_name, frame in scopes:
        for model_name, prefix in models:
            summary_rows.append({
                "scope": scope_name,
                "model": model_name,
                "n": int(len(frame)),
                **{metric: float(frame[f"{prefix}_{metric}"].mean()) for metric in metrics},
            })

    comparison_pairs = [("form", "baseline"), ("form_h2h", "baseline"), ("form_h2h", "form")]
    diff_rows = []
    for scope_name, frame in scopes:
        for challenger, control in comparison_pairs:
            for metric in metrics:
                delta = frame[f"{challenger}_{metric}"] - frame[f"{control}_{metric}"]
                diff_rows.append({
                    "scope": scope_name,
                    "comparison": f"{challenger}_minus_{control}",
                    "metric": metric,
                    "n": int(delta.notna().sum()),
                    "mean_delta": float(delta.mean()),
                    "interpretation": "negative favours challenger; lower scores are better",
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(diff_rows)


def bootstrap_date_clusters(results: pd.DataFrame, reps: int, seed: int = 20261026) -> pd.DataFrame:
    if reps <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    date_groups = {
        str(day): frame for day, frame in results.groupby("match_date_utc", sort=True)
    }
    dates = np.asarray(list(date_groups.keys()), dtype=object)
    if len(dates) < 2:
        return pd.DataFrame()
    comparisons = [("form", "baseline"), ("form_h2h", "baseline"), ("form_h2h", "form")]
    rows = []
    for challenger, control in comparisons:
        for metric in ("brier", "rps", "log_loss"):
            diff_column = f"{challenger}_{metric}"
            control_column = f"{control}_{metric}"
            samples = np.empty(reps, dtype=float)
            for i in range(reps):
                picked = rng.choice(dates, size=len(dates), replace=True)
                vals: list[np.ndarray] = []
                for day in picked:
                    frame = date_groups[str(day)]
                    vals.append((frame[diff_column] - frame[control_column]).to_numpy(dtype=float))
                samples[i] = float(np.concatenate(vals).mean())
            actual = float((results[diff_column] - results[control_column]).mean())
            rows.append({
                "comparison": f"{challenger}_minus_{control}",
                "metric": metric,
                "n_matches": int(len(results)),
                "n_dates": int(len(dates)),
                "mean_delta": actual,
                "ci_95_low": float(np.quantile(samples, 0.025)),
                "ci_95_high": float(np.quantile(samples, 0.975)),
                "bootstrap_reps": int(reps),
                "interpretation": "negative favours challenger; interval crossing zero means uncertainty remains",
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/match_context_features.csv",
                        help="Output from src.build_match_context_features")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Smoke-test limit applied per held-out season")
    parser.add_argument("--min-train-matches", type=int, default=MIN_TRAIN_MATCHES)
    parser.add_argument("--bootstrap-reps", type=int, default=500,
                        help="Date-cluster bootstrap repetitions; use 0 to skip")
    parser.add_argument("--output", default="reports/match_context_backtest.csv",
                        help="Per-match prediction output CSV")
    parser.add_argument("--seasons", nargs="+", default=list(EVALUATION_SEASONS),
                        help="Held-out season keys, e.g. 2023-2024 2024-2025")
    args = parser.parse_args(argv)
    if args.max_dates is not None and args.max_dates < 1:
        parser.error("--max-dates must be >= 1")
    if args.min_train_matches < MIN_TRAIN_MATCHES:
        parser.error(f"--min-train-matches must be >= {MIN_TRAIN_MATCHES}")
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be >= 0")

    seasons = tuple(normalize_season(value) for value in args.seasons)
    if any(season is None for season in seasons):
        parser.error("Invalid --seasons value")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = get_sqlalchemy_engine()
    try:
        all_matches = load_match_data(engine, Path(args.dataset))
    except Exception:
        LOG.exception("Unable to load matches and match-context features")
        return 1
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    missing_seasons = sorted(set(seasons) - set(all_matches["season_key"].dropna().unique()))
    if missing_seasons:
        LOG.error("No database matches found for requested seasons: %s", missing_seasons)
        return 1

    try:
        results, metadata = run_experiment(
            all_matches, tuple(seasons), max_dates=args.max_dates,
            min_train_matches=args.min_train_matches,
        )
        summary, differences = create_summary(results)
        uncertainty = bootstrap_date_clusters(results, args.bootstrap_reps)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_path = output.with_name(output.stem + "_summary.csv")
        diff_path = output.with_name(output.stem + "_differences.csv")
        uncertainty_path = output.with_name(output.stem + "_uncertainty.csv")
        metadata_path = output.with_name(output.stem + "_metadata.csv")
        results.to_csv(output, index=False, float_format="%.10g")
        summary.to_csv(summary_path, index=False, float_format="%.10g")
        differences.to_csv(diff_path, index=False, float_format="%.10g")
        uncertainty.to_csv(uncertainty_path, index=False, float_format="%.10g")
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

        print("\nMATCH-CONTEXT EXPANDING-WINDOW EVALUATION")
        print(f"Held-out seasons: {', '.join(seasons)}")
        print(f"Finished fixtures evaluated (same fixtures for all variants): {metadata['matches_evaluated']:,}")
        print(f"Evaluation dates considered: {metadata['evaluation_dates_considered']:,}")
        print(f"Dates with predictions: {metadata['dates_with_predictions']:,}")
        print(f"Fixtures skipped for insufficient training history: {metadata['skipped_for_insufficient_training']:,}")
        print(f"Predictions using cold-start fallback: {metadata['cold_start_fallback_predictions']:,}")
        print("\nMean scores (lower is better):")
        print(summary.round(5).to_string(index=False))
        print("\nPaired score differences (negative favours the first model; lower is better):")
        print(differences[differences["scope"] == "ALL_HELD_OUT"].round(6).to_string(index=False))
        if not uncertainty.empty:
            print("\nDate-cluster bootstrap 95% intervals for pooled paired score differences:")
            print(uncertainty.round(6).to_string(index=False))
        print("\nReports saved:")
        print(f"  Per-match predictions: {output}")
        print(f"  Summary:               {summary_path}")
        print(f"  Differences:           {diff_path}")
        print(f"  Uncertainty:           {uncertainty_path}")
        print(f"  Metadata:              {metadata_path}")
        print("Research-only: no database writes and no changes to production predictions.")
        return 0
    except Exception:
        LOG.exception("Match-context experiment failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
