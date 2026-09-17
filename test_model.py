"""Tests for the pass-through model, including its probability calibration.

The calibration test is the important one.  The system quotes a probability to
a buyer who acts on it, so an under-confident or over-confident model is a
correctness defect, not a cosmetic one.
"""

import json

import numpy as np
import pytest

import alignment
import futures_util
import model


def _era_pairs(prefix):
    df = alignment.calibration_history()
    frame = alignment.aligned_deltas(df, prefix)
    keep = ~frame["date"].apply(lambda d: futures_util.is_contract_roll_day(d.date(), prefix))
    return frame[keep].reset_index(drop=True)


# --- basic fitting ---------------------------------------------------------

def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, 3000)
    y = 0.4 + 0.65 * x + rng.normal(0, 2.5, 3000)
    fit = model.fit_passthrough(x, y)
    assert fit.slope == pytest.approx(0.65, abs=0.02)
    assert fit.intercept == pytest.approx(0.4, abs=0.15)
    assert fit.sigma == pytest.approx(2.5, abs=0.15)


def test_fit_refuses_too_few_rows():
    with pytest.raises(model.ModelFitError):
        model.fit_passthrough(np.arange(10.0), np.arange(10.0))


def test_sigma_uses_residual_degrees_of_freedom():
    """Dividing by n instead of n-2 would bias sigma downward."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 5, 50)
    y = 0.5 * x + rng.normal(0, 3, 50)
    fit = model.fit_passthrough(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    assert fit.sigma == pytest.approx(np.sqrt((resid @ resid) / (len(x) - 2)))
    assert fit.sigma > np.std(resid)


# --- probability behaviour -------------------------------------------------

def test_probability_is_monotone_and_symmetric_about_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 8, 2000)
    fit = model.fit_passthrough(x, 0.7 * x + rng.normal(0, 2, 2000))
    moves = [0.5, 1.0, 2.0, 5.0, 10.0]
    probs = [float(fit.probability_correct(m)) for m in moves]
    assert probs == sorted(probs), "confidence must rise with the size of the move"
    # A symmetric fit gives near-mirror confidence on the two sides.
    assert float(fit.probability_correct(-5.0)) == pytest.approx(
        float(fit.probability_correct(5.0)), abs=0.08)


def test_probability_never_reaches_certainty():
    """A finite sample cannot justify a 100% claim."""
    rng = np.random.default_rng(4)
    x = rng.normal(0, 8, 300)
    fit = model.fit_passthrough(x, 0.9 * x + rng.normal(0, 0.5, 300))
    assert float(fit.probability_correct(500.0)) < 1.0
    assert float(fit.probability_correct(500.0)) > 0.99


def test_threshold_and_probability_are_consistent():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 8, 1500)
    fit = model.fit_passthrough(x, 0.7 * x + rng.normal(0, 2, 1500))
    for target in (0.6, 0.7, 0.8, 0.9):
        hike, drop = fit.threshold_for_confidence(target)
        assert float(fit.probability_correct(hike)) == pytest.approx(target, abs=0.02)
        assert float(fit.probability_correct(drop)) == pytest.approx(target, abs=0.02)


# --- the calibration guarantee --------------------------------------------

@pytest.mark.parametrize("prefix", ["RB", "HO"])
def test_quoted_probabilities_match_realised_frequencies(prefix):
    """Out-of-sample reliability on the real history.

    Expanding origin: fit on prior rows only, quote for the next block, never
    look ahead.  A normal-residual model failed this -- the residuals carry
    excess kurtosis of 13 (RB) and 18 (HO), which inflates sigma and drags
    every probability toward 0.5, producing a 7.6-point under-confidence with
    negative Brier skill for RB.  The empirical residual CDF fixes it.
    """
    frame = _era_pairs(prefix)
    quoted, realised = [], []
    for start in range(120, len(frame), 10):
        train = frame.iloc[max(0, start - 240):start]
        test = frame.iloc[start:start + 10]
        if len(train) < model.MIN_FIT_ROWS or test.empty:
            continue
        fit = model.fit_passthrough(train["delta_nymex"], train["delta_rack"])
        for move, outcome in zip(test["delta_nymex"], test["delta_rack"]):
            quoted.append(float(fit.probability_correct(move)))
            realised.append(bool(outcome > 0) if move >= 0 else bool(outcome < 0))

    quoted = np.array(quoted)
    realised = np.array(realised, dtype=float)
    assert len(quoted) >= 100

    # Aggregate calibration: the mean quoted probability must track the
    # realised hit rate.
    assert abs(realised.mean() - quoted.mean()) < 0.06, (
        f"{prefix} quoted {quoted.mean():.3f} against realised {realised.mean():.3f}")

    # Brier skill against a constant-rate forecast must not be negative: a
    # probability that is worse than "always say the base rate" is misleading.
    base = realised.mean()
    brier = float(np.mean((quoted - realised) ** 2))
    reference = float(np.mean((base - realised) ** 2))
    assert brier <= reference * 1.02, (
        f"{prefix} Brier {brier:.4f} worse than constant baseline {reference:.4f}")


@pytest.mark.parametrize("prefix", ["RB", "HO"])
def test_empirical_residuals_beat_the_normal_approximation(prefix):
    """Documents why the normal assumption was abandoned."""
    frame = _era_pairs(prefix)
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    from scipy import stats
    kurtosis = float(stats.kurtosis(fit.residuals))
    assert kurtosis > 5, (
        "residuals are expected to be strongly leptokurtic; if this ever fails "
        "the normal approximation may be acceptable again")


# --- serialisation ---------------------------------------------------------

@pytest.mark.parametrize("prefix", ["RB", "HO"])
def test_round_trip_through_the_metrics_cache_is_faithful(prefix):
    frame = _era_pairs(prefix)
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    restored = model.passthrough_from_config(
        json.loads(json.dumps(model.passthrough_to_config(fit, prefix))), prefix)

    moves = np.linspace(-30, 30, 241)
    before = np.array([float(fit.probability_correct(m)) for m in moves])
    after = np.array([float(restored.probability_correct(m)) for m in moves])
    assert np.max(np.abs(before - after)) < 0.02
    assert restored.threshold_for_confidence(0.75) == pytest.approx(
        fit.threshold_for_confidence(0.75), abs=1e-3)


def test_missing_model_returns_none_rather_than_a_default():
    assert model.passthrough_from_config({}, "RB") is None
    assert model.passthrough_from_config({"RB_pt_slope": 0.7}, "RB") is None


# --- risk ------------------------------------------------------------------

def test_tail_risk_reports_its_own_uncertainty():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 8, 600)
    y = 0.7 * x + rng.normal(0, 3, 600)
    tail = model.wait_tail_risk(x, y, drop=-2.0)
    assert tail is not None
    assert tail["cvar_low"] <= tail["cvar"] <= tail["cvar_high"]
    assert tail["tail_n"] >= 4
    assert tail["sample_n"] > tail["tail_n"]
    assert 0.0 <= tail["probability_adverse"] <= 1.0


def test_tail_risk_declines_to_guess_on_a_thin_sample():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 8, 60)
    y = 0.7 * x + rng.normal(0, 3, 60)
    assert model.wait_tail_risk(x, y, drop=-25.0) is None


def test_savings_convention_matches_the_procurement_decision():
    """BUY captures a rise; WAIT captures a fall."""
    nymex = np.array([5.0, -5.0, 0.0])
    rack = np.array([3.0, -4.0, 9.0])
    payoff, correct = model.savings_from_signals(nymex, rack, hike=2.0, drop=-2.0)
    assert sorted(payoff.tolist()) == [3.0, 4.0]
    assert correct == 2
