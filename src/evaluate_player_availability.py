#!/usr/bin/env python3
"""First leakage-aware test of player lineup strength in Dixon-Coles.

The first player feature is the attacking-strength delta of each confirmed XI
relative to that club's recent starting-XI mean. A single shared coefficient is
estimated using 2022-23 outcomes and then frozen for evaluation on 2023-24 and
2024-25. Team attack/defence strengths are refit chronologically before each
evaluation date, using only matches on earlier dates.

This is a research/backtest script. It never writes to PostgreSQL and never
changes locked production predictions.

Run from repository root:
    python -m src.evaluate_player_availability --max-dates 2
    python -m src.evaluate_player_availability \
        --output reports/player_availability_backtest.csv
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

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
from src.models.dixon_coles import (
    DixonColesModel,
    outcome_probs,
    score_matrix,
)

LOG = logging.getLogger("evaluate_player_availability")

PLAYER_TRAIN_SEASON = "2022-2023"
EVALUATION_SEASONS = ("2023-2024", "2024-2025")
FEATURE_HOME = "home_xi_attack_delta_vs_recent_team_mean"
FEATURE_AWAY = "away_xi_attack_delta_vs_recent_team_mean"
FEATURE_READY = "lineup_delta_features_ready"
TIME_DECAY = 0.001
RIDGE = 1.0
THETA_RIDGE = 2.0
MAX_GOALS = 9
MIN_TRAIN_MATCHES = 200
THETA_BOUNDS = (-1.5, 1.5)
RHO_BOUNDS = (-0.3, 0.3)

MATCHES_SQL = text("""
    SELECT m.id AS match_id,
           m.season,
           m.matchweek,
           m.kickoff_utc AS database_kickoff_utc,
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
    ORDER BY m.kickoff_utc ASC, m.id ASC
""")


# --------------------------------------------------------------------------- I/O

def parse_flag(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def normalize_season(value) -> str:
    text_value = str(value).strip()
    if len(text_value) == 7 and text_value[4] == "-":
        start, end = text_value.split("-", 1)
        if len(start) == 4 and len(end) == 2 and start.isdigit() and end.isdigit():
            return f"{start}-{int(start[:2]) * 100 + int(end):04d}"
    return text_value


def load_all_matches(engine, dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Joined player dataset not found: {dataset_path}. Run "
            "python -m src.prepare_player_model_dataset first."
        )
    player_rows = pd.read_csv(dataset_path, low_memory=False)
    needed = {
        "match_id", "season", "match_date_utc", "kickoff_utc",
        FEATURE_HOME, FEATURE_AWAY, FEATURE_READY,
    }
    missing = sorted(needed - set(player_rows.columns))
    if missing:
        raise ValueError(f"Player model dataset missing required columns: {missing}")

    player_rows["match_id"] = pd.to_numeric(player_rows["match_id"], errors="coerce")
    player_rows = player_rows.dropna(subset=["match_id"]).copy()
    player_rows["match_id"] = player_rows["match_id"].astype(int)
    if player_rows["match_id"].duplicated().any():
        raise ValueError("Joined player dataset contains duplicate match_id values")
    player_rows["feature_kickoff_utc"] = pd.to_datetime(
        player_rows["kickoff_utc"], utc=True, errors="coerce"
    )
    player_rows["feature_match_date_utc"] = pd.to_datetime(
        player_rows["match_date_utc"], utc=True, errors="coerce"
    )
    player_rows["player_feature_ready"] = player_rows[FEATURE_READY].map(parse_flag)
    for column in (FEATURE_HOME, FEATURE_AWAY):
        player_rows[column] = pd.to_numeric(player_rows[column], errors="coerce")
    player_rows["player_feature_ready"] &= (
        np.isfinite(player_rows[FEATURE_HOME]) & np.isfinite(player_rows[FEATURE_AWAY])
    )
    player_rows = player_rows[[
        "match_id", "feature_kickoff_utc", "feature_match_date_utc",
        FEATURE_HOME, FEATURE_AWAY, "player_feature_ready",
    ]]

    with engine.connect() as connection:
        matches = pd.read_sql_query(MATCHES_SQL, connection)
    if matches.empty:
        raise RuntimeError("No completed, scored matches found in PostgreSQL")
    if pd.to_numeric(matches["match_id"], errors="coerce").isna().any():
        raise ValueError("Database returned invalid match IDs")
    matches["match_id"] = matches["match_id"].astype(int)
    matches["season_key"] = matches["season"].map(normalize_season)
    matches["database_kickoff_utc"] = pd.to_datetime(
        matches["database_kickoff_utc"], utc=True, errors="coerce"
    )
    matches = matches.merge(player_rows, how="left", on="match_id", validate="one_to_one")

    # The FPL archive's match date is authoritative for fixtures with player
    # features because the feature builder uses that exact UTC date for its
    # same-day leakage guard. Elsewhere we use the DB fixture timestamp.
    feature_date = matches["feature_match_date_utc"].dt.date
    db_date = matches["database_kickoff_utc"].dt.date
    date_mismatch = feature_date.notna() & db_date.notna() & (feature_date != db_date)
    matches["kickoff_utc"] = matches["feature_kickoff_utc"].combine_first(
        matches["database_kickoff_utc"]
    )
    matches["_match_date"] = matches["kickoff_utc"].dt.date
    matches = matches.dropna(subset=[
        "kickoff_utc", "_match_date", "home_team_id", "away_team_id",
        "home_goals", "away_goals",
    ]).copy()
    matches["home_team_id"] = matches["home_team_id"].astype(int)
    matches["away_team_id"] = matches["away_team_id"].astype(int)
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["player_feature_ready"] = matches["player_feature_ready"].fillna(False).astype(bool)
    matches["player_x_home_raw"] = matches[FEATURE_HOME]
    matches["player_x_away_raw"] = matches[FEATURE_AWAY]
    matches["_feature_date_mismatch"] = date_mismatch
    return matches.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def fit_feature_scaler(matches: pd.DataFrame) -> tuple[float, float, int]:
    """Freeze the scaling using only player-feature training season 2022-23."""
    sample = matches[
        (matches["season_key"] == PLAYER_TRAIN_SEASON)
        & matches["player_feature_ready"]
    ]
    values = pd.concat([
        sample["player_x_home_raw"], sample["player_x_away_raw"],
    ], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 100:
        raise ValueError(
            f"Only {len(values)} team-lineup delta values are available in "
            f"{PLAYER_TRAIN_SEASON}; need at least 100 to fit a stable scale."
        )
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("Player lineup-delta feature has near-zero variance")
    return mean, std, len(sample)


def add_scaled_features(matches: pd.DataFrame, mean: float, std: float) -> pd.DataFrame:
    data = matches.copy()
    active = data["player_feature_ready"].to_numpy(dtype=bool)
    home_raw = pd.to_numeric(data["player_x_home_raw"], errors="coerce").to_numpy(dtype=float)
    away_raw = pd.to_numeric(data["player_x_away_raw"], errors="coerce").to_numpy(dtype=float)
    home_z = np.zeros(len(data), dtype=float)
    away_z = np.zeros(len(data), dtype=float)
    home_z[active] = (home_raw[active] - mean) / std
    away_z[active] = (away_raw[active] - mean) / std
    data["player_x_home_z"] = home_z
    data["player_x_away_z"] = away_z
    return data


# -------------------------------------------------------------------- rates/metrics

def league_average_rates(train: pd.DataFrame, reference_date: pd.Timestamp) -> tuple[float, float]:
    kickoff = pd.to_datetime(train["kickoff_utc"], utc=True)
    age_days = ((reference_date - kickoff).dt.total_seconds() / 86400.0).clip(lower=0).to_numpy()
    weights = np.exp(-TIME_DECAY * age_days)
    lam = float(np.average(train["home_goals"].to_numpy(dtype=float), weights=weights))
    mu = float(np.average(train["away_goals"].to_numpy(dtype=float), weights=weights))
    if not np.isfinite([lam, mu]).all() or lam <= 0 or mu <= 0:
        raise ValueError(f"Invalid baseline rates: home={lam}, away={mu}")
    return lam, mu


def validate_distribution(matrix, probs) -> None:
    matrix = np.asarray(matrix, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(probs).all():
        raise ValueError("Model returned non-finite probability values")
    if (matrix < 0).any() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Model returned probabilities outside their valid ranges")
    if abs(float(matrix.sum()) - 1.0) > 1e-8 or abs(float(probs.sum()) - 1.0) > 1e-8:
        raise ValueError("Model probabilities do not sum to one")
    derived = outcome_probs(matrix)
    if np.max(np.abs(np.asarray(derived) - probs)) > 1e-8:
        raise ValueError("1X2 probabilities disagree with the score matrix")


def score_probs(probs, outcome: str) -> dict[str, float]:
    h, d, a = map(float, probs)
    scores = {
        "brier": calculate_brier_score(h, d, a, outcome),
        "rps": calculate_rps(h, d, a, outcome),
        "log_loss": calculate_log_loss(h, d, a, outcome),
    }
    if not np.isfinite(list(scores.values())).all():
        raise ValueError(f"Non-finite scores for probabilities={probs}, outcome={outcome}")
    return scores


# ---------------------------------------------------------------- Dixon-Coles research fitter

class ResearchDixonColes:
    """Dixon-Coles fitter with an optional fixed or estimated lineup-delta slope.

    Parameterisation matches the project model: sum-zero attack strengths,
    unconstrained defence strengths, a home-advantage term, and rho. It uses the
    same time-decay/ridge settings and score-matrix builder as the production
    fitter, plus one optional regularised shared lineup coefficient.
    """

    def __init__(self, ridge: float = RIDGE, theta_ridge: float = THETA_RIDGE):
        self.ridge = ridge
        self.theta_ridge = theta_ridge
        self.team_index: dict[int, int] = {}
        self.attacks = np.array([])
        self.defences = np.array([])
        self.home_advantage = 0.0
        self.rho = 0.0
        self.theta = 0.0
        self.converged = False
        self.n_train = 0

    @staticmethod
    def _unpack(params: np.ndarray, n_teams: int, estimate_theta: bool):
        n_attack_free = n_teams - 1
        attack_free = params[:n_attack_free]
        attacks = np.concatenate([attack_free, [-float(np.sum(attack_free))]])
        def_start = n_attack_free
        defences = params[def_start:def_start + n_teams]
        gamma_idx = def_start + n_teams
        gamma = float(params[gamma_idx])
        rho = float(params[gamma_idx + 1])
        theta = float(params[gamma_idx + 2]) if estimate_theta else None
        return attacks, defences, gamma, rho, theta, gamma_idx

    def fit(self, train: pd.DataFrame, reference_date: pd.Timestamp,
            fixed_theta: float | None = None, estimate_theta: bool = False):
        if train.empty or len(train) < MIN_TRAIN_MATCHES:
            raise ValueError(f"Need at least {MIN_TRAIN_MATCHES} training matches; got {len(train)}")
        if estimate_theta == (fixed_theta is not None):
            raise ValueError("Specify exactly one: estimate_theta=True or fixed_theta=<value>")
        if estimate_theta:
            fixed_theta = None
        self.theta = 0.0 if fixed_theta is None else float(fixed_theta)

        train = train.copy()
        train["kickoff_utc"] = pd.to_datetime(train["kickoff_utc"], utc=True, errors="coerce")
        train = train.dropna(subset=["kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals"])
        if len(train) < MIN_TRAIN_MATCHES:
            raise ValueError(f"Only {len(train)} clean training matches remain")
        team_ids = sorted(set(train["home_team_id"].astype(int)) | set(train["away_team_id"].astype(int)))
        if len(team_ids) < 2:
            raise ValueError("At least two teams are required")
        self.team_index = {team_id: i for i, team_id in enumerate(team_ids)}
        h_idx = train["home_team_id"].astype(int).map(self.team_index).to_numpy(dtype=int)
        a_idx = train["away_team_id"].astype(int).map(self.team_index).to_numpy(dtype=int)
        home_goals = train["home_goals"].to_numpy(dtype=float)
        away_goals = train["away_goals"].to_numpy(dtype=float)
        kickoff = pd.to_datetime(train["kickoff_utc"], utc=True)
        age_days = ((reference_date - kickoff).dt.total_seconds() / 86400.0).clip(lower=0.0).to_numpy()
        weights = np.exp(-TIME_DECAY * age_days)
        x_home = train.get("player_x_home_z", pd.Series(0.0, index=train.index)).fillna(0.0).to_numpy(dtype=float)
        x_away = train.get("player_x_away_z", pd.Series(0.0, index=train.index)).fillna(0.0).to_numpy(dtype=float)
        n_teams = len(team_ids)

        # Initialize near league scoring averages to improve optimizer stability.
        home_mean = max(float(np.average(home_goals, weights=weights)), 0.05)
        away_mean = max(float(np.average(away_goals, weights=weights)), 0.05)
        initial = np.zeros((n_teams - 1) + n_teams + 2 + int(estimate_theta), dtype=float)
        initial[(n_teams - 1):(n_teams - 1 + n_teams)] = math.log(away_mean)
        initial[(n_teams - 1 + n_teams)] = math.log(home_mean / away_mean)
        # rho starts at zero. A free theta also starts at zero.
        bounds = (
            [(None, None)] * ((n_teams - 1) + n_teams)
            + [(-1.5, 1.5), RHO_BOUNDS]
            + ([THETA_BOUNDS] if estimate_theta else [])
        )

        def objective(params):
            attacks, defences, gamma, rho, theta_fit, gamma_idx = self._unpack(params, n_teams, estimate_theta)
            theta = float(theta_fit) if estimate_theta else self.theta
            eta_h = attacks[h_idx] + defences[a_idx] + gamma + theta * x_home
            eta_a = attacks[a_idx] + defences[h_idx] + theta * x_away
            if (not np.isfinite(eta_h).all() or not np.isfinite(eta_a).all()
                    or np.max(eta_h) > 15 or np.max(eta_a) > 15
                    or np.min(eta_h) < -15 or np.min(eta_a) < -15):
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

            # Derivatives of the negative weighted log likelihood plus ridge.
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
            if estimate_theta:
                theta_grad = -float(np.sum(weighted_gh * x_home + weighted_ga * x_away)) + self.theta_ridge * theta
                grad_parts.append(np.array([theta_grad]))
            penalty = 0.5 * self.ridge * (float(np.sum(attacks ** 2)) + float(np.sum(defences ** 2)))
            if estimate_theta:
                penalty += 0.5 * self.theta_ridge * theta ** 2
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
                f"Dixon-Coles optimisation failed: {result.message}; "
                f"iterations={result.nit}; objective={result.fun}"
            )
        attacks, defences, gamma, rho, theta_fit, _ = self._unpack(result.x, n_teams, estimate_theta)
        self.attacks = attacks
        self.defences = defences
        self.home_advantage = gamma
        self.rho = rho
        if estimate_theta:
            self.theta = float(theta_fit)
        self.converged = True
        self.n_train = len(train)
        self.optimization_message = str(result.message)
        return self

    def predict_probs(self, home_team_id: int, away_team_id: int,
                      x_home_z: float, x_away_z: float,
                      max_goals: int = MAX_GOALS):
        if home_team_id not in self.team_index or away_team_id not in self.team_index:
            return None
        hi = self.team_index[int(home_team_id)]
        ai = self.team_index[int(away_team_id)]
        lam = math.exp(float(self.attacks[hi] + self.defences[ai] + self.home_advantage + self.theta * x_home_z))
        mu = math.exp(float(self.attacks[ai] + self.defences[hi] + self.theta * x_away_z))
        matrix = score_matrix(lam, mu, self.rho, max_goals)
        probs = outcome_probs(matrix)
        return tuple(map(float, probs)), lam, mu, matrix


def scoreline_metrics(matrix, outcome: str) -> dict[str, float]:
    probs = tuple(map(float, outcome_probs(matrix)))
    validate_distribution(matrix, probs)
    return score_probs(probs, outcome)


def bootstrap_date_clusters(results: pd.DataFrame, metric: str, reps: int, seed: int = 20261024):
    """Paired date-cluster bootstrap for player model minus raw Dixon-Coles."""
    difference = results[f"player_{metric}"].to_numpy() - results[f"dc_{metric}"].to_numpy()
    cluster = (results["season"].astype(str) + "|" + results["match_date_utc"].astype(str)).to_numpy()
    unique = np.unique(cluster)
    members = {value: np.flatnonzero(cluster == value) for value in unique}
    by_cluster = np.array([float(np.mean(difference[members[value]])) for value in unique])
    sizes = np.array([len(members[value]) for value in unique], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for index in range(reps):
        sampled = rng.integers(0, len(unique), size=len(unique))
        # Pool fixture differences, not cluster means, to preserve the estimand.
        num = sum(float(np.sum(difference[members[unique[j]]])) for j in sampled)
        den = sum(len(members[unique[j]]) for j in sampled)
        draws[index] = num / den if den else np.nan
    return {
        "metric": metric,
        "n": len(results),
        "mean_player_minus_raw": float(np.mean(difference)),
        "ci95_lower": float(np.nanquantile(draws, 0.025)),
        "ci95_upper": float(np.nanquantile(draws, 0.975)),
        "bootstrap_reps": reps,
        "cluster_unit": "season_utc_match_date",
    }


def run_experiment(all_matches: pd.DataFrame, scaler_mean: float, scaler_std: float,
                   max_dates: int | None = None, min_train_matches: int = MIN_TRAIN_MATCHES):
    # Estimate the one shared player coefficient using 2022-23 only, then freeze it.
    target_start = all_matches.loc[
        all_matches["season_key"].eq("2023-2024"), "_match_date"
    ].min()
    if pd.isna(target_start):
        raise ValueError("No 2023-2024 completed fixtures found for the held-out evaluation")
    theta_train = all_matches[all_matches["_match_date"] < target_start].copy()
    if len(theta_train) < min_train_matches:
        raise ValueError(f"Only {len(theta_train)} matches before 2023-24 player-coefficient training")
    theta_train_feature_n = int((theta_train["season_key"].eq(PLAYER_TRAIN_SEASON) & theta_train["player_feature_ready"]).sum())
    theta_reference = pd.Timestamp(target_start).tz_localize("UTC")
    theta_model = ResearchDixonColes().fit(
        theta_train, reference_date=theta_reference, estimate_theta=True
    )
    theta = float(theta_model.theta)

    print("PLAYER-AVAILABILITY MODEL EXPERIMENT")
    print(f"Player coefficient training season: {PLAYER_TRAIN_SEASON}")
    print(f"Player-feature-ready training fixtures: {theta_train_feature_n}")
    print(f"Training matches available at coefficient fit: {len(theta_train):,}")
    print(f"Frozen lineup-delta coefficient (theta): {theta:.6f}")
    print(f"Multiplicative scoring-rate effect per +1 SD lineup delta: {math.exp(theta):.4f}x")
    print("Held-out seasons: 2023-2024, 2024-2025")
    print("Feature: current XI attacking contribution relative to recent team lineup mean")
    print("Note: this is an observational association, not a causal injury effect.")

    records: list[dict] = []
    excluded_feature_rows = {season: 0 for season in EVALUATION_SEASONS}
    cold_start_matches = 0
    insufficient_history = 0
    evaluation_dates_total = 0

    for season in EVALUATION_SEASONS:
        season_rows = all_matches[all_matches["season_key"].eq(season)].copy()
        if season_rows.empty:
            raise ValueError(f"No completed matches available for target season {season}")
        all_dates = sorted(season_rows["_match_date"].unique())
        if max_dates is not None:
            all_dates = all_dates[:max_dates]
        evaluation_dates_total += len(all_dates)
        for eval_date in all_dates:
            train = all_matches[all_matches["_match_date"] < eval_date].copy()
            test_day_all = season_rows[season_rows["_match_date"] == eval_date]
            test_day = test_day_all[test_day_all["player_feature_ready"]].copy()
            excluded_feature_rows[season] += len(test_day_all) - len(test_day)
            if test_day.empty:
                continue
            if len(train) < min_train_matches:
                insufficient_history += len(test_day)
                continue
            reference = pd.Timestamp(eval_date).tz_localize("UTC")

            # Fit the unmodified control from the project's production class.
            raw_model = DixonColesModel(time_decay=TIME_DECAY, ridge=RIDGE)
            raw_model.fit(train, reference_date=reference)
            # The player-aware model keeps theta learned on 2022-23 frozen, but
            # refits team strengths/rho using only outcomes from earlier dates.
            player_model = ResearchDixonColes().fit(
                train, reference_date=reference, fixed_theta=theta
            )
            base_lam, base_mu = league_average_rates(train, reference)
            base_matrix = score_matrix(base_lam, base_mu, 0.0, MAX_GOALS)
            base_probs = tuple(map(float, outcome_probs(base_matrix)))
            validate_distribution(base_matrix, base_probs)

            trained_team_ids = (
                set(train["home_team_id"].astype(int))
                | set(train["away_team_id"].astype(int))
            )
            for match in test_day.itertuples(index=False):
                outcome = outcome_from_goals(match.home_goals, match.away_goals)
                x_home = float(match.player_x_home_z) if np.isfinite(match.player_x_home_z) else 0.0
                x_away = float(match.player_x_away_z) if np.isfinite(match.player_x_away_z) else 0.0
                source = "dixon_coles"
                known_in_both = (
                    int(match.home_team_id) in trained_team_ids
                    and int(match.away_team_id) in trained_team_ids
                    and int(match.home_team_id) in player_model.team_index
                    and int(match.away_team_id) in player_model.team_index
                )
                if not known_in_both:
                    # Same cold-start policy for both models so the player feature
                    # experiment stays paired and does not silently lose promoted clubs.
                    dc_probs = base_probs
                    dc_lam, dc_mu = base_lam, base_mu
                    player_probs = base_probs
                    player_lam, player_mu = base_lam, base_mu
                    source = "baseline_cold_start"
                    cold_start_matches += 1
                else:
                    p_h, p_d, p_a, dc_lam, dc_mu, dc_matrix = raw_model.predict_probs(
                        int(match.home_team_id), int(match.away_team_id), MAX_GOALS
                    )
                    dc_probs = (float(p_h), float(p_d), float(p_a))
                    validate_distribution(dc_matrix, dc_probs)
                    predicted = player_model.predict_probs(
                        int(match.home_team_id), int(match.away_team_id), x_home, x_away, MAX_GOALS
                    )
                    if predicted is None:
                        raise RuntimeError("Player model unexpectedly returned no prediction")
                    player_probs, player_lam, player_mu, player_matrix = predicted
                    validate_distribution(player_matrix, player_probs)

                dc_scores = score_probs(dc_probs, outcome)
                player_scores = score_probs(player_probs, outcome)
                base_scores = score_probs(base_probs, outcome)
                records.append({
                    "match_id": int(match.match_id),
                    "season": season,
                    "match_date_utc": str(eval_date),
                    "matchweek": match.matchweek,
                    "home_name": match.home_name,
                    "away_name": match.away_name,
                    "home_goals": int(match.home_goals),
                    "away_goals": int(match.away_goals),
                    "actual_outcome": outcome,
                    "n_train": int(len(train)),
                    "train_cutoff_date_exclusive": str(eval_date),
                    "player_feature_ready": True,
                    "player_x_home_raw": float(match.player_x_home_raw),
                    "player_x_away_raw": float(match.player_x_away_raw),
                    "player_x_home_z": x_home,
                    "player_x_away_z": x_away,
                    "player_theta_frozen": theta,
                    "prediction_source": source,
                    "dc_p_home": dc_probs[0], "dc_p_draw": dc_probs[1], "dc_p_away": dc_probs[2],
                    "dc_expected_home_goals": float(dc_lam), "dc_expected_away_goals": float(dc_mu),
                    "dc_brier": dc_scores["brier"], "dc_rps": dc_scores["rps"], "dc_log_loss": dc_scores["log_loss"],
                    "player_p_home": player_probs[0], "player_p_draw": player_probs[1], "player_p_away": player_probs[2],
                    "player_expected_home_goals": float(player_lam), "player_expected_away_goals": float(player_mu),
                    "player_brier": player_scores["brier"], "player_rps": player_scores["rps"], "player_log_loss": player_scores["log_loss"],
                    "baseline_p_home": base_probs[0], "baseline_p_draw": base_probs[1], "baseline_p_away": base_probs[2],
                    "baseline_expected_home_goals": base_lam, "baseline_expected_away_goals": base_mu,
                    "baseline_brier": base_scores["brier"], "baseline_rps": base_scores["rps"], "baseline_log_loss": base_scores["log_loss"],
                })
            LOG.info("Evaluated %s %s: training=%d, feature-ready=%d", season, eval_date, len(train), len(test_day))

    results = pd.DataFrame.from_records(records)
    if results.empty:
        raise RuntimeError("No player-feature-ready matches were evaluated")
    metadata = {
        "theta": theta,
        "theta_training_matches": len(theta_train),
        "theta_training_feature_matches": theta_train_feature_n,
        "evaluation_dates_considered": evaluation_dates_total,
        "matches_evaluated": len(results),
        "excluded_feature_unready": sum(excluded_feature_rows.values()),
        "excluded_feature_unready_by_season": excluded_feature_rows,
        "cold_start_baseline_fallbacks": cold_start_matches,
        "insufficient_history_skipped": insufficient_history,
        "feature_scaler_mean": scaler_mean,
        "feature_scaler_std": scaler_std,
        "database_vs_fpl_date_mismatches": int(all_matches["_feature_date_mismatch"].sum()),
        "min_train_matches": min_train_matches,
    }
    return results, metadata


def create_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [
        ("dixon_coles_v2", "dc"),
        ("dixon_coles_player_v1", "player"),
        ("baseline_poisson_v1", "baseline"),
    ]
    metric_names = ("brier", "rps", "log_loss")
    summary_rows = []
    for scope, frame in [(s, results[results["season"] == s]) for s in EVALUATION_SEASONS] + [("ALL_HELD_OUT", results)]:
        for model_name, prefix in models:
            summary_rows.append({
                "scope": scope, "model": model_name, "n": len(frame),
                **{metric: float(frame[f"{prefix}_{metric}"].mean()) for metric in metric_names},
            })
    summary = pd.DataFrame(summary_rows)
    differences = []
    for metric in metric_names:
        delta = results[f"player_{metric}"] - results[f"dc_{metric}"]
        differences.append({"scope": "ALL_HELD_OUT", "metric": metric, "n": len(results),
                            "mean_player_minus_raw": float(delta.mean())})
        for season in EVALUATION_SEASONS:
            values = results.loc[results["season"] == season, f"player_{metric}"] - results.loc[results["season"] == season, f"dc_{metric}"]
            differences.append({"scope": season, "metric": metric, "n": len(values),
                                "mean_player_minus_raw": float(values.mean()) if len(values) else np.nan})
    return summary, pd.DataFrame(differences)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/player_model_dataset.csv",
                        help="joined output from prepare_player_model_dataset.py")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="smoke-test limit applied per held-out season")
    parser.add_argument("--min-train-matches", type=int, default=MIN_TRAIN_MATCHES)
    parser.add_argument("--bootstrap-reps", type=int, default=2000,
                        help="date-cluster bootstrap repetitions; use 0 to skip")
    parser.add_argument("--output", default="reports/player_availability_backtest.csv",
                        help="path for paired per-match results CSV")
    args = parser.parse_args(argv)
    if args.max_dates is not None and args.max_dates < 1:
        parser.error("--max-dates must be >= 1")
    if args.min_train_matches < MIN_TRAIN_MATCHES:
        parser.error(f"--min-train-matches must be >= {MIN_TRAIN_MATCHES}")
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be >= 0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset_path = Path(args.dataset)
    engine = get_sqlalchemy_engine()
    try:
        all_matches = load_all_matches(engine, dataset_path)
    except Exception:
        logging.exception("Unable to load completed matches and player features")
        engine.dispose()
        return 1
    finally:
        # The engine is read-only; no transaction is left open.
        try:
            engine.dispose()
        except Exception:
            pass

    try:
        scaler_mean, scaler_std, scaler_n = fit_feature_scaler(all_matches)
        all_matches = add_scaled_features(all_matches, scaler_mean, scaler_std)
        results, metadata = run_experiment(
            all_matches, scaler_mean, scaler_std, max_dates=args.max_dates,
            min_train_matches=args.min_train_matches,
        )
        summary, differences = create_summary(results)
        if args.bootstrap_reps:
            ci = pd.DataFrame([
                bootstrap_date_clusters(results, metric, args.bootstrap_reps)
                for metric in ("brier", "rps", "log_loss")
            ])
        else:
            ci = pd.DataFrame()

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_path = output.with_name(output.stem + "_summary.csv")
        diff_path = output.with_name(output.stem + "_differences.csv")
        ci_path = output.with_name(output.stem + "_uncertainty.csv")
        metadata_path = output.with_name(output.stem + "_metadata.csv")
        results.to_csv(output, index=False, float_format="%.10g")
        summary.to_csv(summary_path, index=False, float_format="%.10g")
        differences.to_csv(diff_path, index=False, float_format="%.10g")
        ci.to_csv(ci_path, index=False, float_format="%.10g")
        pd.DataFrame([metadata | {"feature_scaler_n_team_values": scaler_n}]).to_csv(
            metadata_path, index=False, float_format="%.10g"
        )

        print("\nPLAYER-AVAILABILITY EXPANDING-WINDOW EVALUATION")
        print(f"Frozen theta trained on {PLAYER_TRAIN_SEASON}: {metadata['theta']:.6f}")
        print(f"Scoring-rate multiplier per +1 SD lineup delta: {math.exp(metadata['theta']):.4f}x")
        print(f"Held-out seasons: {', '.join(EVALUATION_SEASONS)}")
        print(f"Fixtures evaluated (paired, lineup features ready): {metadata['matches_evaluated']}")
        print(f"Evaluation dates considered: {metadata['evaluation_dates_considered']}")
        print(f"Fixtures excluded for missing lineup-delta features: {metadata['excluded_feature_unready']}")
        print(f"Cold-start baseline fallbacks within paired sample: {metadata['cold_start_baseline_fallbacks']}")
        print(f"Database vs FPL kickoff-date mismatches: {metadata['database_vs_fpl_date_mismatches']}")
        print("\nMean scores (lower is better):")
        print(summary.round(5).to_string(index=False))
        print("\nPaired difference (player model minus raw Dixon-Coles; negative favours player model):")
        print(differences.round(6).to_string(index=False))
        if not ci.empty:
            print("\nDate-cluster bootstrap intervals for pooled held-out paired differences:")
            print(ci.round(6).to_string(index=False))
        print("\nReports saved:")
        print(f"  Per-match predictions: {output}")
        print(f"  Summary:               {summary_path}")
        print(f"  Differences:           {diff_path}")
        print(f"  Uncertainty:           {ci_path}")
        print(f"  Metadata:              {metadata_path}")
        print("Research-only: nothing written to PostgreSQL or production prediction tables.")
        return 0
    except Exception:
        LOG.exception("Player-availability experiment failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
