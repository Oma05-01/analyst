"""
Dixon-Coles model: independent Poisson goals for each side plus the Dixon-Coles
correction for the four low scores (0-0, 0-1, 1-0, 1-1).

    log(lambda) = attack[home] + defence[away] + gamma      (home goals)
    log(mu)     = attack[away] + defence[home]              (away goals)

`defence` is a *weakness* parameter: higher means the team concedes more.
Attack ratings are centred to sum to zero (that removes the one free direction).

Compared with v1:
  * analytic gradient + L-BFGS-B, and a convergence check (fit raises on failure)
  * ridge penalty on attack / defence (a Gaussian prior that shrinks ratings of
    teams with little data toward the league average)
  * rho is bounded
  * unknown teams raise UnknownTeamError instead of returning invented numbers
  * decay can be measured from an explicit reference date (needed for backtests)

Any change to what this file computes should ship under a NEW model_version.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

TAU_FLOOR = 1e-10


class UnknownTeamError(Exception):
    """A team has no fitted rating (for example a newly promoted club)."""


class ModelNotFittedError(RuntimeError):
    pass


class FitError(RuntimeError):
    pass


def score_matrix(lam, mu, rho=0.0, max_goals=9):
    """(max_goals+1) x (max_goals+1) probability matrix, rows = home goals, cols = away goals."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho
    matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid score matrix (lambda={lam}, mu={mu}, rho={rho})")
    return matrix / total   # renormalise: the grid is truncated at max_goals


def outcome_probs(matrix):
    """Home win / draw / away win probabilities read off the SAME matrix that is displayed."""
    return (
        float(np.tril(matrix, -1).sum()),
        float(np.trace(matrix)),
        float(np.triu(matrix, 1).sum()),
    )


class DixonColesModel:
    RHO_BOUNDS = (-0.3, 0.3)

    def __init__(self, time_decay=0.001, ridge=1.0, min_matches=50):
        self.time_decay = time_decay      # per day: half-life = ln(2) / time_decay days
        self.ridge = ridge
        self.min_matches = min_matches
        self.params = None
        self.teams = []
        self.team_indices = {}
        self.fit_info = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, matches_df, reference_date=None, init_params=None):
        required = {"home_team_id", "away_team_id", "home_goals", "away_goals"}
        missing = required - set(matches_df.columns)
        if missing:
            raise ValueError(f"matches_df is missing columns: {sorted(missing)}")
        if len(matches_df) < self.min_matches:
            raise FitError(f"Need at least {self.min_matches} matches to fit, got {len(matches_df)}")
        if matches_df[["home_goals", "away_goals"]].isna().any().any():
            raise ValueError("matches_df contains missing goals")

        teams = sorted(set(matches_df["home_team_id"]).union(matches_df["away_team_id"]))
        if len(teams) < 2:
            raise FitError("Need at least two teams to fit")
        n = len(teams)
        self.teams = [int(t) for t in teams]
        self.team_indices = {t: i for i, t in enumerate(self.teams)}

        h = np.array([self.team_indices[int(t)] for t in matches_df["home_team_id"]])
        a = np.array([self.team_indices[int(t)] for t in matches_df["away_team_id"]])
        x = matches_df["home_goals"].to_numpy().astype(int)
        y = matches_df["away_goals"].to_numpy().astype(int)
        if (x < 0).any() or (y < 0).any():
            raise ValueError("Negative goals in matches_df")
        xf, yf = x.astype(float), y.astype(float)
        m = len(x)

        if "kickoff_utc" in matches_df.columns:
            kickoff = pd.to_datetime(matches_df["kickoff_utc"], utc=True)
            if reference_date is None:
                reference = kickoff.max()
            else:
                reference = pd.Timestamp(reference_date)
                reference = reference.tz_localize("UTC") if reference.tzinfo is None else reference.tz_convert("UTC")
            days = ((reference - kickoff).dt.total_seconds() / 86400.0).to_numpy()
            weights = np.exp(-self.time_decay * np.clip(days, 0.0, None))
        else:
            weights = np.ones(m)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        ridge = self.ridge

        def nll_and_grad(params):
            att_raw = params[:n]
            def_ = params[n:2 * n]
            gamma = params[2 * n]
            rho = params[2 * n + 1]
            att = att_raw - att_raw.mean()

            log_lam = att[h] + def_[a] + gamma
            log_mu = att[a] + def_[h]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)

            tau = np.ones(m)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            valid = tau > TAU_FLOOR
            tau_safe = np.where(valid, tau, TAU_FLOOR)

            # Poisson log-pmf without the constant log(k!) term (it does not depend on params)
            ll = np.log(tau_safe) + xf * log_lam - lam + yf * log_mu - mu
            penalty = ridge * (np.sum(att ** 2) + np.sum((def_ - def_.mean()) ** 2))
            nll = -np.sum(weights * ll) + penalty

            # ---- gradient
            d_lam = xf - lam
            d_mu = yf - mu
            dt_lam = np.zeros(m)
            dt_mu = np.zeros(m)
            dt_rho = np.zeros(m)
            dt_lam[m00] = -lam[m00] * mu[m00] * rho
            dt_mu[m00] = -lam[m00] * mu[m00] * rho
            dt_rho[m00] = -lam[m00] * mu[m00]
            dt_lam[m01] = lam[m01] * rho
            dt_rho[m01] = lam[m01]
            dt_mu[m10] = mu[m10] * rho
            dt_rho[m10] = mu[m10]
            dt_rho[m11] = -1.0
            inv = np.where(valid, 1.0 / tau_safe, 0.0)
            g_lam = weights * (d_lam + dt_lam * inv)
            g_mu = weights * (d_mu + dt_mu * inv)

            g_att = np.bincount(h, g_lam, n) + np.bincount(a, g_mu, n)
            g_def = np.bincount(a, g_lam, n) + np.bincount(h, g_mu, n)
            grad_att_c = -g_att + 2.0 * ridge * att
            grad_att = grad_att_c - grad_att_c.mean()      # chain rule through the centring
            grad_def = -g_def + 2.0 * ridge * (def_ - def_.mean())
            grad = np.concatenate([
                grad_att, grad_def,
                [-g_lam.sum()],
                [-np.sum(weights * dt_rho * inv)],
            ])
            return nll, grad

        if init_params is not None and len(init_params) == 2 * n + 2:
            start = np.asarray(init_params, dtype=float)
        else:
            start = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        bounds = [(None, None)] * (2 * n + 1) + [self.RHO_BOUNDS]

        res = minimize(nll_and_grad, start, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 2000})
        if not res.success or not np.all(np.isfinite(res.x)):
            raise FitError(f"Dixon-Coles fit did not converge: {res.message}")

        fitted = res.x.copy()
        fitted[:n] -= fitted[:n].mean()    # keep stored ratings consistent with the objective
        self.params = fitted
        self.fit_info = {
            "n_matches": int(m),
            "n_teams": int(n),
            "iterations": int(res.nit),
            "final_nll": float(res.fun),
            "converged": True,
        }
        return self

    # --------------------------------------------------------------- prediction
    def _require_fitted(self):
        if self.params is None:
            raise ModelNotFittedError("Model has not been fitted")

    @property
    def gamma(self):
        self._require_fitted()
        return float(self.params[2 * len(self.teams)])

    @property
    def rho(self):
        self._require_fitted()
        return float(self.params[2 * len(self.teams) + 1])

    def predict_probs(self, home_team_id, away_team_id, max_goals=9):
        """Returns (p_home, p_draw, p_away, lambda, mu, matrix). Never invents numbers."""
        self._require_fitted()
        for team_id in (home_team_id, away_team_id):
            if int(team_id) not in self.team_indices:
                raise UnknownTeamError(int(team_id))

        n = len(self.teams)
        att = self.params[:n]
        def_ = self.params[n:2 * n]
        h = self.team_indices[int(home_team_id)]
        a = self.team_indices[int(away_team_id)]

        lam = float(np.exp(att[h] + def_[a] + self.gamma))
        mu = float(np.exp(att[a] + def_[h]))
        matrix = score_matrix(lam, mu, self.rho, max_goals)
        p_home, p_draw, p_away = outcome_probs(matrix)
        return p_home, p_draw, p_away, lam, mu, matrix

    # ------------------------------------------------------------------- extras
    def describe(self):
        """JSON-serialisable summary to store alongside each prediction."""
        self._require_fitted()
        return {
            "time_decay": self.time_decay,
            "ridge": self.ridge,
            "gamma": self.gamma,
            "rho": self.rho,
            "n_teams": len(self.teams),
            "n_matches": self.fit_info.get("n_matches"),
        }

    def ratings_frame(self):
        self._require_fitted()
        n = len(self.teams)
        return pd.DataFrame({
            "team_id": self.teams,
            "attack": self.params[:n],
            "defence_weakness": self.params[n:2 * n],
        }).sort_values("attack", ascending=False).reset_index(drop=True)