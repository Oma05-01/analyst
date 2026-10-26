import pytest

from src.metrics import (calculate_brier_score, calculate_log_loss, calculate_rps,
                         outcome_from_goals)

P = (0.5, 0.3, 0.2)


def test_home_win():
    assert calculate_brier_score(*P, "HOME") == pytest.approx(0.38)
    assert calculate_rps(*P, "HOME") == pytest.approx(0.145)
    assert calculate_log_loss(*P, "HOME") == pytest.approx(0.693147, abs=1e-5)


def test_draw():
    assert calculate_brier_score(*P, "DRAW") == pytest.approx(0.78)
    assert calculate_rps(*P, "DRAW") == pytest.approx(0.145)
    assert calculate_log_loss(*P, "DRAW") == pytest.approx(1.203973, abs=1e-5)


def test_away_win():
    assert calculate_brier_score(*P, "AWAY") == pytest.approx(0.98)
    assert calculate_rps(*P, "AWAY") == pytest.approx(0.445)
    assert calculate_log_loss(*P, "AWAY") == pytest.approx(1.609438, abs=1e-5)


@pytest.mark.parametrize("bad", ["Home", "H", "", None, "away"])
def test_invalid_outcome_raises(bad):
    for fn in (calculate_brier_score, calculate_rps, calculate_log_loss):
        with pytest.raises(ValueError):
            fn(*P, bad)


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError):
        calculate_rps(0.7, 0.7, 0.2, "HOME")        # does not sum to 1
    with pytest.raises(ValueError):
        calculate_rps(1.2, -0.1, -0.1, "HOME")      # out of range
    with pytest.raises(ValueError):
        calculate_rps(float("nan"), 0.5, 0.5, "HOME")


def test_outcome_from_goals():
    assert outcome_from_goals(2, 1) == "HOME"
    assert outcome_from_goals(1, 1) == "DRAW"
    assert outcome_from_goals(0, 3) == "AWAY"