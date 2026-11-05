"""Causality and paired-comparison guards for the read-only research tool."""

import numpy as np
import pandas as pd
import pytest

import model
from scratch import compare_adaptive_models as comparison


def synthetic_frame(n=375):
    rng = np.random.default_rng(123)
    x = rng.normal(0, 4, n)
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "delta_nymex": x,
        "delta_rack": 0.7 * x + rng.normal(0, 0.5, n),
    })


def test_current_candidate_reuses_production_fit():
    frame = synthetic_frame()
    expected = model.fit_passthrough(frame.tail(360)["delta_nymex"],
                                     frame.tail(360)["delta_rack"])
    actual = comparison.fit_candidate(frame, "current_360")
    assert actual.slope == expected.slope
    assert actual.intercept == expected.intercept
    np.testing.assert_array_equal(actual.residuals, expected.residuals)


def test_test_and_future_outcomes_never_affect_their_prediction():
    frame = synthetic_frame()
    original = comparison.evaluate(frame, 0.75, 0.65, 0.6)
    changed = frame.copy()
    changed.loc[361:, "delta_rack"] += 100
    alternative = comparison.evaluate(changed, 0.75, 0.65, 0.6)
    first = frame.loc[361, "date"]
    cols = ["candidate", "expected", "probability", "probability_up", "direction"]
    pd.testing.assert_frame_equal(
        original[original["date"] == first][cols].reset_index(drop=True),
        alternative[alternative["date"] == first][cols].reset_index(drop=True))
    assert (original["train_end"] < original["date"]).all()
    assert original.groupby("candidate")["date"].nunique().eq(len(frame) - 361).all()


def test_purge_row_does_not_enter_training():
    frame = synthetic_frame()
    original = comparison.evaluate(frame, 0.75, 0.65, 0.6)
    frame.loc[360, "delta_rack"] += 100
    changed = comparison.evaluate(frame, 0.75, 0.65, 0.6)
    first = original["date"].min()
    np.testing.assert_array_equal(
        original.loc[original["date"] == first, "expected"],
        changed.loc[changed["date"] == first, "expected"])


def test_weighted_coefficients_adapt_to_recent_response():
    frame = synthetic_frame(360)
    frame["delta_rack"] = np.where(np.arange(360) < 270, 0.5, 1.5) * frame["delta_nymex"]
    ordinary = comparison.fit_candidate(frame, "current_360")
    weighted = comparison.fit_candidate(frame, "recent_weighted_360")
    assert abs(weighted.slope - 1.5) < abs(ordinary.slope - 1.5)


def test_paired_baseline_improvements_are_zero():
    records = comparison.evaluate(synthetic_frame(), 0.75, 0.65, 0.6)
    summary = comparison.summarize(records)
    current = summary["current_360"]
    assert current["mse_improvement_vs_current"] == 0
    assert current["payoff_improvement_cents_per_session"] == 0
    assert current["mse_improvement_95_interval"] == [0, 0]
    assert current["payoff_improvement_95_interval"] == [0, 0]
    assert len({item["sessions"] for item in summary.values()}) == 1


def test_comparison_rejects_unpaired_dates():
    records = comparison.evaluate(synthetic_frame(), 0.75, 0.65, 0.6)
    index = records[records["candidate"] == "short_180"].index[-1]
    with pytest.raises(ValueError, match="dates do not match"):
        comparison.summarize(records.drop(index))


def test_future_append_leaves_prior_forecasts_unchanged():
    frame = synthetic_frame(380)
    before = comparison.evaluate(frame.iloc[:375], 0.75, 0.65, 0.6)
    after = comparison.evaluate(frame, 0.75, 0.65, 0.6)
    pd.testing.assert_frame_equal(before, after[after["date"] <= before["date"].max()])
