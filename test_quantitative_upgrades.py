import csv

import pandas as pd
import pytest
import numpy as np

import main
import model
import record_execution
import validate_data
import weekly_report
from calibration_artifacts import append_calibration_artifact


def _live_frame():
    return pd.DataFrame({
        "timestamp_dt": pd.to_datetime([
            "2026-09-21T19:00:00Z", "2026-09-22T19:00:00Z",
            "2026-09-23T19:00:00Z",
        ], utc=True).tz_convert("America/Chicago"),
        "commodity": ["RB", "RB", "HO"],
        "predicted_direction": ["HIKE", "FLAT", "DROP"],
        "actual_move": [2.0, -4.0, 1.0],
        "nymex_move_cents": [3.0, -2.0, -3.0],
        "is_correct": [True, False, False],
        "savings_cents": [2.0, 0.0, -1.0],
        "conviction_label": [
            "High confidence (90%) | p=0.9000",
            "Moderate confidence (70%) | p=0.7000",
            "Moderate confidence (75%) | p=0.7500",
        ],
    })


def test_live_descriptive_summary_is_useful_below_thirty_rows():
    result = weekly_report.live_descriptive_summary(_live_frame())
    assert result["All"]["alerts"] == 2
    assert result["All"]["correct"] == 1
    assert result["All"]["precision"] == pytest.approx(0.5)
    assert result["All"]["precision_low"] < 0.5 < result["All"]["precision_high"]
    assert result["All"]["avg_savings"] == pytest.approx(0.5)
    assert result["All"]["max_adverse"] == pytest.approx(-1.0)
    assert result["All"]["brier"] == pytest.approx((0.1 ** 2 + 0.75 ** 2) / 2)
    assert result["All"]["calibration_gap"] == pytest.approx(0.5 - 0.825)
    assert result["RB"]["alerts"] == 1
    assert result["HO"]["alerts"] == 1


def test_policy_benchmark_uses_identical_rows_and_keeps_flat_as_no_action():
    result = weekly_report.paired_policy_benchmark(
        _live_frame(), bootstrap=200, block_length=2, seed=4)
    assert result["accepted"] == 2
    assert result["rejected"] == 1
    assert result["model_total"] == pytest.approx(1.0)
    assert result["model_avg_active"] == pytest.approx(0.5)
    assert result["sign_total"] == pytest.approx(5.0)
    assert result["incremental_total"] == pytest.approx(-4.0)
    assert result["incremental_ci_low"] <= result["incremental_total"] <= result["incremental_ci_high"]


def test_execution_writer_is_idempotent_and_summary_uses_only_priced_rows(tmp_path):
    path = tmp_path / "execution.csv"
    base = {column: "" for column in record_execution.EXECUTION_COLUMNS}
    base.update({
        "record_id": "a" * 32, "signal_date": "2026-09-22", "commodity": "RB",
        "recommendation": "HIKE", "executed_action": "DISPATCHED_SAME_DAY",
        "gallons": "8500", "load_date": "2026-09-22",
        "price_paid_per_gallon": "3.40", "counterfactual_price_per_gallon": "3.44",
        "realized_savings_dollars": "340.00", "recorded_at": "2026-09-22T14:00:00-05:00",
    })
    record_execution.write_execution(base, path=str(path))
    with pytest.raises(ValueError, match="already recorded"):
        record_execution.write_execution(base, path=str(path))
    second = dict(base, record_id="b" * 32, commodity="HO", recommendation="DROP", gallons="7500",
                  price_paid_per_gallon="", counterfactual_price_per_gallon="",
                  realized_savings_dollars="")
    record_execution.write_execution(second, path=str(path))
    summary = weekly_report.execution_summary(str(path))
    assert summary == {
        "records": 2, "priced": 1, "gallons": 16000.0,
        "realized_dollars": 340.0, "unpriced": 1,
    }

    correction = dict(base, record_id="c" * 32,
                      counterfactual_price_per_gallon="3.45",
                      realized_savings_dollars="425.00")
    record_execution.write_execution(
        correction, path=str(path), supersedes="a" * 32)
    corrected = weekly_report.execution_summary(str(path))
    assert corrected["records"] == 2
    assert corrected["realized_dollars"] == pytest.approx(425.0)
    validate_data.validate_execution_log(str(path))


def test_live_recommendation_rejects_backfill_and_ambiguity(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [
        {"timestamp": "2026-09-22T14:00:00-05:00", "commodity": "RB",
         "prediction_source": "backfill", "predicted_direction": "DROP"},
        {"timestamp": "2026-09-22T14:35:00-05:00", "commodity": "RB",
         "prediction_source": "live", "predicted_direction": "HIKE"},
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    assert record_execution.live_recommendation(
        "2026-09-22", "RB", str(path)) == "HIKE"
    with pytest.raises(ValueError, match="found 0"):
        record_execution.live_recommendation("2026-09-22", "HO", str(path))


def test_live_signal_applies_economic_gate_and_threshold_stability(monkeypatch):
    rng = np.random.default_rng(44)
    x = rng.normal(0, 5, 400)
    fit = model.fit_passthrough(x, 0.7 * x + rng.normal(0, 1, 400))
    cfg = {
        "RB_HIKE_THRESHOLD_CENTS": 1.0,
        "RB_DROP_THRESHOLD_CENTS": -1.0,
        "RB_LEAN_HIKE_CENTS": 0.5,
        "RB_LEAN_DROP_CENTS": -0.5,
        "RB_nymex_daily_std": 5.0,
        "RB_threshold_ci_hike_high": 2.5,
        "RB_threshold_ci_drop_low": -2.5,
        "RB_BUY_INCREMENTAL_COST_CENTS_PER_GAL": 3.0,
        "RB_WAIT_INCREMENTAL_COST_CENTS_PER_GAL": 0.0,
        "TRUCK_GALLONS": 8500,
        **model.passthrough_to_config(fit, "RB"),
    }
    monkeypatch.setattr(main, "APP_CONFIG", cfg)
    monkeypatch.setattr(main, "append_prediction_log", lambda *args, **kwargs: True)
    data = {
        "yesterday_close": 2.00,
        "current_price": 2.03,
        "baseline_source": "settlement_provenance_verified",
        "schwab_symbol": "/RBV26",
        "baseline_schwab_symbol": "/RBV26",
        "data_source": "schwab",
        "settlement_snapshot": {
            "price": 2.03, "schwab_symbol": "/RBV26",
            "source": "schwab", "captured_at": "2026-09-23T13:31:00-05:00",
        },
    }
    signal = main.build_rack_signal(
        "RB", data, pd.Timestamp("2026-09-23 14:35", tz="America/Chicago").to_pydatetime())
    assert signal["action"] == "NO_EDGE"
    assert signal["label"] == "No economic edge"
    assert signal["economic_value"]["net_edge_cents"] < 0
    assert signal["threshold_stability"]["stable"] is True
    assert "not expected to cover" in signal["text"]
    assert "No operating cost is configured" not in signal["risk_text"]

    cfg["RB_BUY_INCREMENTAL_COST_CENTS_PER_GAL"] = 0.0
    signal = main.build_rack_signal(
        "RB", data, pd.Timestamp("2026-09-23 14:35", tz="America/Chicago").to_pydatetime())
    assert signal["action"] == "BUY_NOW"
    assert signal["economic_value"]["net_edge_cents"] > 0
    assert "No operating cost is configured" in signal["risk_text"]


def test_future_cache_activates_exact_session_artifact_without_losing_policy(
        monkeypatch, tmp_path):
    path = tmp_path / "runs.jsonl"
    artifact = {
        "artifact_schema_version": 2,
        "effective_session": "2026-09-24",
        "training_start": "2025-01-02",
        "training_end": "2026-09-22",
        "purge_rows": 1,
        "source_history_hash": "a" * 64,
        "source_row_count": 300,
        "candidate_grid_version": "v2-test",
        "candidate_grid": {"window": 360},
        "objective": "test",
        "smoothing_input": {"window": 360},
        "prior_artifact_id": "bootstrap_config",
        "calibration": {"RB_HIKE_THRESHOLD_CENTS": 2.0},
        "generated_at": "2026-09-23T18:00:00-05:00",
    }
    written, _ = append_calibration_artifact(str(path), artifact)
    cfg = {
        "CALIBRATION_EFFECTIVE_SESSION": "2026-09-25",
        "RB_HIKE_THRESHOLD_CENTS": 9.0,
        "RB_threshold_ci_hike_high": 9.5,
        "RB_BUY_INCREMENTAL_COST_CENTS_PER_GAL": 1.25,
    }
    monkeypatch.setattr(main, "APP_CONFIG", cfg)
    activated = main.activate_calibration_for_session(
        pd.Timestamp("2026-09-24 14:00", tz="America/Chicago").to_pydatetime(),
        artifact_path=str(path))
    assert activated == "2026-09-24"
    assert cfg["RB_HIKE_THRESHOLD_CENTS"] == 2.0
    assert "RB_threshold_ci_hike_high" not in cfg
    assert cfg["RB_BUY_INCREMENTAL_COST_CENTS_PER_GAL"] == 1.25
    assert cfg["CALIBRATION_ARTIFACT_ID"] == written["artifact_id"]
