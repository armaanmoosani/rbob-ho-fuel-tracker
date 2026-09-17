"""End-to-end smoke test of the calibration engine's components."""

import numpy as np
import pandas as pd
import pytest

import alignment
import backtest
import model


def _synthetic(rows=400, slope=0.7, seed=5):
    """Synthetic history inside the alignment-verified era."""
    dates = pd.date_range(alignment.CALIBRATION_ERA_START, periods=rows, freq="B")
    rng = np.random.default_rng(seed)
    step = rng.normal(0, 0.03, rows)
    nymex = 2.0 + step.cumsum()
    rack = np.empty(rows)
    rack[0] = 2.1
    rack[1:] = rack[0] + (slope * step[1:] + rng.normal(0, 0.008, rows - 1)).cumsum()
    return pd.DataFrame({
        "date": dates,
        "nymex_rb": nymex, "nymex_ho": nymex,
        "rack_u": rack, "rack_p": rack + 0.1, "rack_d": rack,
    })


def test_pairs_are_same_session_and_roll_free():
    df = _synthetic()
    frame = backtest.training_pairs(df, "RB")
    assert not frame.empty
    assert {"date", "delta_nymex", "delta_rack"} <= set(frame.columns)
    # Every pair must span exactly one observed session of each series.
    assert frame["delta_nymex"].notna().all()
    assert frame["delta_rack"].notna().all()
    from futures_util import is_contract_roll_day
    assert not any(is_contract_roll_day(d.date(), "RB") for d in frame["date"])


def test_fit_recovers_the_true_pass_through():
    frame = backtest.training_pairs(_synthetic(slope=0.7), "RB")
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    assert 0.6 < fit.slope < 0.8
    assert fit.r2 > 0.8


def test_thresholds_clear_the_noise_floor_and_bracket_zero():
    frame = backtest.training_pairs(_synthetic(), "RB")
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    floor = backtest.DEFAULTS["SNAPSHOT_NOISE_FLOOR_CENTS"]
    hike, drop = model.apply_noise_floor(
        *fit.threshold_for_confidence(backtest.DEFAULTS["TARGET_SIGNAL_CONFIDENCE"]),
        floor)
    assert hike >= floor
    assert drop <= -floor
    assert drop < 0 < hike


def test_walk_forward_reports_out_of_sample_performance():
    frame = backtest.training_pairs(_synthetic(), "RB")
    result = backtest.walk_forward_evaluation(frame, backtest.DEFAULTS)
    assert len(result["folds"]) == backtest.EVAL_FOLDS
    assert result["alerts"] > 0
    assert 0.0 <= result["precision"] <= 1.0
    # A 0.7 pass-through with small noise should be comfortably profitable.
    assert result["precision"] > 0.6


def test_calibration_populates_every_published_key():
    df = _synthetic()
    cfg, message, evaluation = backtest.calibrate(df, "RB", dict(backtest.DEFAULTS))
    for key in ("RB_HIKE_THRESHOLD_CENTS", "RB_DROP_THRESHOLD_CENTS",
                "RB_pt_slope", "RB_pt_intercept", "RB_pt_residual_quantiles",
                "RB_oos_precision", "RB_oos_alerts", "RB_oos_window",
                "RB_insample_precision", "RB_wait_cvar_status"):
        assert key in cfg, f"calibration did not publish {key}"
    assert cfg["RB_oos_alerts"] == evaluation["alerts"]
    assert "OOS prec=" in message


def test_calibration_raises_rather_than_silently_falling_back():
    """A geometry that does not fit must fail loudly.

    The superseded engine returned a -9999 sentinel and reverted to a hardcoded
    (120, 15, 85), producing plausible thresholds with no warning.
    """
    with pytest.raises(model.ModelFitError):
        backtest.calibrate(_synthetic(rows=60), "RB", dict(backtest.DEFAULTS))


def test_load_config_returns_policy_defaults():
    cfg = backtest.load_config()
    assert isinstance(cfg, dict)
    for key in ("TARGET_SIGNAL_CONFIDENCE", "SNAPSHOT_NOISE_FLOOR_CENTS",
                "ROLLING_WINDOW_DAYS"):
        assert key in cfg
    assert 0.5 < cfg["TARGET_SIGNAL_CONFIDENCE"] < 1.0
