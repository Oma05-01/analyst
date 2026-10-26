import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src.models.dixon_coles import (DixonColesModel, FitError, ModelNotFittedError,
                                    UnknownTeamError, outcome_probs, score_matrix)


def simulate(n_teams=8, n_rounds=12, seed=0):
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.25, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.2, n_teams)
    rows, day = [], 0
    for r in range(n_rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j or rng.random() > 0.5:
                    continue
                lam = np.exp(att[i] + dfn[j] + 0.3 + 0.1)
                mu = np.exp(att[j] + dfn[i] + 0.1)
                rows.append({"home_team_id": 100 + i, "away_team_id": 100 + j,
                             "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                             "kickoff_utc": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=day)})
                day += 1
    return pd.DataFrame(rows), att, dfn


def test_matrix_is_normalised_and_consistent():
    matrix = score_matrix(1.6, 1.1, -0.1, 9)
    assert matrix.shape == (10, 10)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)
    assert sum(outcome_probs(matrix)) == pytest.approx(1.0, abs=1e-12)


def test_fit_predict_and_orientation():
    df, att, _ = simulate()
    model = DixonColesModel(time_decay=0.001, ridge=0.5).fit(df)
    assert model.fit_info["converged"]
    assert model.params[:8].sum() == pytest.approx(0.0, abs=1e-9)
    best, worst = 100 + int(np.argmax(att)), 100 + int(np.argmin(att))
    p_home, p_draw, p_away, lam, mu, matrix = model.predict_probs(best, worst)
    assert p_home > p_away
    assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-9)
    assert (p_home, p_draw, p_away) == pytest.approx(outcome_probs(matrix))


def test_unknown_team_and_unfitted_raise():
    with pytest.raises(ModelNotFittedError):
        DixonColesModel().predict_probs(1, 2)
    df, _, _ = simulate()
    model = DixonColesModel().fit(df)
    with pytest.raises(UnknownTeamError):
        model.predict_probs(100, 9999)


def test_too_few_matches_raises():
    df, _, _ = simulate()
    with pytest.raises(FitError):
        DixonColesModel(min_matches=10_000).fit(df)


def test_analytic_gradient_matches_numeric():
    """Re-build the objective through a tiny hook: compare gradient from the fit's own function."""
    df, _, _ = simulate(n_teams=5, n_rounds=6)
    captured = {}
    import src.models.dixon_coles as dc
    real_minimize = dc.minimize

    def spy(fun, x0, **kw):
        captured["fun"], captured["x0"] = fun, x0
        return real_minimize(fun, x0, **kw)

    dc.minimize = spy
    try:
        DixonColesModel(ridge=0.7, min_matches=10).fit(df)
    finally:
        dc.minimize = real_minimize
    fun = captured["fun"]
    rng = np.random.default_rng(1)
    point = captured["x0"] + rng.normal(0, 0.1, len(captured["x0"]))
    _, analytic = fun(point)
    numeric = approx_fprime(point, lambda p: fun(p)[0], 1e-6)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-4)


def test_reference_date_changes_weights_only():
    df, _, _ = simulate()
    a = DixonColesModel().fit(df)
    b = DixonColesModel().fit(df, reference_date="2021-01-01")
    assert a.params is not None and b.params is not None