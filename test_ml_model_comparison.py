"""Decision-time availability and evaluation guards for offline ML research."""

import numpy as np
import pandas as pd
import pytest

from scratch import compare_ml_models as research


@pytest.fixture(autouse=True)
def small_geometry(monkeypatch):
    monkeypatch.setattr(research, "WINDOW", 80)
    monkeypatch.setattr(research, "BASE_MIN", 40)
    monkeypatch.setattr(research, "CORRECTION_MIN", 20)
    monkeypatch.setattr(research, "CALIBRATION_WINDOW", 20)
    monkeypatch.setattr(research, "EVALUATION_START", 85)
    monkeypatch.setitem(research.HYPERPARAMETERS, "boosting", {
        **research.HYPERPARAMETERS["boosting"], "n_estimators": 5})


def synthetic_data(n=100):
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2023-01-02", periods=n + 1)
    x = rng.normal(0, 4, n)
    y = 0.7 * x + rng.normal(0, 0.8, n)
    history = pd.DataFrame({
        "date": dates,
        "nymex_rb": 3 + np.r_[0, np.cumsum(x)] / 100,
        "rack_u": 3.2 + np.r_[0, np.cumsum(y)] / 100,
    })
    frame = pd.DataFrame({
        "date": dates[1:], "prior_date": dates[:-1],
        "delta_nymex": x, "delta_rack": y,
    })
    return frame, history


def test_current_and_future_outcomes_cannot_change_current_forecasts():
    frame, history = synthetic_data()
    original = research.evaluate(frame, history, "RB")
    cutoff = 90
    changed = frame.copy()
    changed.loc[cutoff:, "delta_rack"] += 50
    altered_history = history.copy()
    affected = altered_history["date"] >= frame.iloc[cutoff]["date"]
    altered_history.loc[affected, "rack_u"] += np.arange(1, affected.sum() + 1) * 0.5
    new = research.evaluate(changed, altered_history, "RB")
    cols = ["date", "candidate", "point_prediction", "p_up", "p_down",
            "interval_low", "interval_high", "direction"]
    pd.testing.assert_frame_equal(
        original.loc[original["date"] <= frame.iloc[cutoff]["date"], cols],
        new.loc[new["date"] <= frame.iloc[cutoff]["date"], cols])


def test_future_append_preserves_existing_predictions():
    frame, history = synthetic_data(105)
    before = research.evaluate(frame.iloc[:100], history.iloc[:101], "RB")
    after = research.evaluate(frame, history, "RB")
    pd.testing.assert_frame_equal(before, after[after["date"] <= before["date"].max()])


def test_correction_targets_are_purged_prequential_baseline_errors(monkeypatch):
    frame, history = synthetic_data()
    baseline, _ = research.baseline_forecasts(frame)
    true_errors = frame["delta_rack"].to_numpy() - baseline
    original_fit = research.fit_checked
    observed = []

    def checking_fit(estimator, x, y):
        if isinstance(x, pd.DataFrame):
            indices = x.index.to_numpy()
            np.testing.assert_array_equal(y, true_errors[indices])
            observed.append(indices[-1])
        return original_fit(estimator, x, y)

    monkeypatch.setattr(research, "fit_checked", checking_fit)
    records = research.evaluate(frame, history, "RB")
    assert observed
    assert (records["train_end"] < records["date"]).all()
    assert (records["calibration_end"] < records["date"]).all()
    for row in records.itertuples():
        index = frame.index[frame["date"] == row.date][0]
        assert row.train_end == frame.iloc[index - research.PURGE - 1]["date"]
        assert row.calibration_end <= row.train_end
    # Two correction fits per decision, and neither can train on the purge row.
    assert observed[-2:] == [len(frame) - research.PURGE - 2] * 2


def test_prior_basis_uses_previous_session_including_weekends():
    frame, history = synthetic_data()
    baseline, _ = research.baseline_forecasts(frame)
    features = research.causal_features(frame, history, "RB", baseline)
    monday = frame.index[(frame["date"].dt.weekday == 0) & (frame.index > 50)][0]
    prior = history.set_index("date").loc[frame.iloc[monday]["prior_date"]]
    assert frame.iloc[monday]["prior_date"].weekday() == 4
    assert features.loc[monday, "prior_basis_cents"] == pytest.approx(
        (prior["rack_u"] - prior["nymex_rb"]) * 100)


def test_prior_error_does_not_slide_across_missing_dates():
    frame, history = synthetic_data()
    frame = frame.drop(index=79).reset_index(drop=True)
    baseline, _ = research.baseline_forecasts(frame)
    features = research.causal_features(frame, history, "RB", baseline)
    row = frame.index[frame["date"] == history.iloc[81]["date"]][0]
    assert features.loc[row, "prior_base_error_missing"] == 1
    assert features.loc[row, "prior_base_error_cents"] == 0
    # The raw prior posting still exists even though that date has no eligible pair.
    assert features.loc[row, "prior_basis_cents"] == pytest.approx(
        (history.iloc[80]["rack_u"] - history.iloc[80]["nymex_rb"]) * 100)


def test_volatility_does_not_include_current_move():
    frame, history = synthetic_data()
    baseline, _ = research.baseline_forecasts(frame)
    before = research.causal_features(frame, history, "RB", baseline)
    frame.loc[90, "delta_nymex"] += 100
    after = research.causal_features(frame, history, "RB", baseline)
    assert before.loc[90, "prior_20_move_std_cents"] == after.loc[90, "prior_20_move_std_cents"]
    assert before.loc[91, "prior_20_move_std_cents"] != after.loc[91, "prior_20_move_std_cents"]


def test_empirical_probabilities_allow_zero_move_mass():
    up, down, low, high = research.residual_distribution(0.0, [-1.0, 0.0, 1.0])
    assert up == pytest.approx(0.4)
    assert down == pytest.approx(0.4)
    assert up + down < 1
    assert (low, high) == pytest.approx((-0.8, 0.8))


def test_summary_uses_identical_dates_and_zero_control_differences():
    frame, history = synthetic_data()
    records = research.evaluate(frame, history, "RB")
    result = research.summarize(records, bootstrap=100)
    baseline = result["production_ols_360"]["vs_production"]
    adapter = result["prequential_ols_360"]["vs_same_adapter_ols"]
    for control in (baseline, adapter):
        assert control["mae_improvement_cents"] == 0
        assert control["mae_improvement_95_interval"] == [0, 0]
        assert control["brier_improvement"] == 0
        assert control["payoff_improvement_cents_per_session"] == 0
    assert len({item["sessions"] for item in result.values()}) == 1
    index = records[records["candidate"] == "ridge_residual"].index[-1]
    with pytest.raises(ValueError, match="identical unique dates"):
        research.summarize(records.drop(index), bootstrap=100)


def test_probability_policy_requires_confidence_and_noise_floor():
    assert research.probability_action(1.0, 0.9, 0.1, 0.5, 0.6, 0.75, 0.65) == (0, "FLAT")
    assert research.probability_action(1.0, 0.6, 0.4, 2.0, 0.6, 0.75, 0.65) == (0, "FLAT")
    assert research.probability_action(-1.0, 0.1, 0.9, 2.0, 0.6, 0.75, 0.65) == (-1, "FULL")
    assert research.probability_action(1.0, 0.7, 0.3, 2.0, 0.6, 0.75, 0.65) == (1, "LEAN")


def test_cli_refuses_production_data_output(monkeypatch):
    monkeypatch.setattr("sys.argv", ["compare_ml_models.py", "--output-dir", str(research.ROOT / "data")])
    with pytest.raises(SystemExit) as exc:
        research.main()
    assert exc.value.code == 2


def test_paired_blocks_preserve_cross_candidate_dependence():
    values = np.column_stack([np.arange(40), -np.arange(40), np.ones(40)])
    intervals = research.paired_block_intervals(values, 5, resamples=1000)
    np.testing.assert_allclose(intervals[0], -intervals[1, ::-1])
    np.testing.assert_array_equal(intervals[2], [1.0, 1.0])
    adjusted = research.paired_block_intervals(values, 5, resamples=1000)
    unadjusted = research.paired_block_intervals(values, 5, resamples=1000, confidence=0.95)
    assert adjusted[0, 0] <= unadjusted[0, 0]
    assert adjusted[0, 1] >= unadjusted[0, 1]


def test_primary_loss_family_has_exactly_three_challengers():
    frame, history = synthetic_data()
    records = research.evaluate(frame, history, "RB")
    output = research.primary_inference(records, resamples=1000)
    assert set(output) == {"huber_360", "ridge_residual", "boosted_residual"}
    for result in output.values():
        assert set(result["bonferroni_6_comparison_intervals"]) == {"5", "10", "20"}
