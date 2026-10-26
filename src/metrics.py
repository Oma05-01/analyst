"""
Scoring rules for three-way football outcome probabilities (HOME, DRAW, AWAY).

Conventions (state these on the dashboard):
  * Brier score: SUM over the three classes of (p - outcome)^2. Range 0..2.
    Some references average over the classes instead, which divides by 3.
  * RPS (ranked probability score): outcomes are ordered HOME < DRAW < AWAY and
    the cumulative squared error is divided by (classes - 1) = 2. Range 0..1.
  * Log loss: natural log of the probability given to what actually happened.
Lower is better for all three.
"""
import math

import numpy as np

VALID_OUTCOMES = ("HOME", "DRAW", "AWAY")
PROB_SUM_TOLERANCE = 1e-3   # old rows may have been stored rounded to 4 decimals
LOG_LOSS_EPS = 1e-15


def outcome_from_goals(home_goals, away_goals):
    home, away = int(home_goals), int(away_goals)
    if home > away:
        return "HOME"
    if home < away:
        return "AWAY"
    return "DRAW"


def _validate(p_home, p_draw, p_away, actual_outcome):
    if actual_outcome not in VALID_OUTCOMES:
        raise ValueError(f"Unknown outcome: {actual_outcome!r} (expected one of {VALID_OUTCOMES})")
    probs = (float(p_home), float(p_draw), float(p_away))
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs):
        raise ValueError(f"Invalid probabilities: {probs}")
    if abs(sum(probs) - 1.0) > PROB_SUM_TOLERANCE:
        raise ValueError(f"Probabilities do not sum to 1: {probs} (sum={sum(probs)})")
    return probs


def calculate_brier_score(p_home, p_draw, p_away, actual_outcome):
    p_home, p_draw, p_away = _validate(p_home, p_draw, p_away, actual_outcome)
    y_true = np.array([
        1.0 if actual_outcome == "HOME" else 0.0,
        1.0 if actual_outcome == "DRAW" else 0.0,
        1.0 if actual_outcome == "AWAY" else 0.0,
    ])
    y_pred = np.array([p_home, p_draw, p_away])
    return float(np.sum((y_pred - y_true) ** 2))


def calculate_rps(p_home, p_draw, p_away, actual_outcome):
    p_home, p_draw, p_away = _validate(p_home, p_draw, p_away, actual_outcome)
    cum_1 = p_home
    cum_2 = p_home + p_draw
    if actual_outcome == "HOME":
        actual_1, actual_2 = 1.0, 1.0
    elif actual_outcome == "DRAW":
        actual_1, actual_2 = 0.0, 1.0
    else:  # AWAY
        actual_1, actual_2 = 0.0, 0.0
    return float(0.5 * ((cum_1 - actual_1) ** 2 + (cum_2 - actual_2) ** 2))


def calculate_log_loss(p_home, p_draw, p_away, actual_outcome):
    p_home, p_draw, p_away = _validate(p_home, p_draw, p_away, actual_outcome)
    p = {"HOME": p_home, "DRAW": p_draw, "AWAY": p_away}[actual_outcome]
    return float(-np.log(max(p, LOG_LOSS_EPS)))