"""Prospective capture, immutable provenance, scoring, and isolation tests."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import alignment
import backtest
import main
import shadow_model
import shadow_trial as trial
import weekly_report
from scratch import compare_ml_models as research


def at(day="2026-11-05", hour=14, minute=36):
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=trial.TZ)


@pytest.fixture(scope="module")
def historical_pairs():
    history = alignment.calibration_history(alignment.load_history())
    history = history[history["date"] <= pd.Timestamp("2026-11-04")]
    return backtest.training_pairs(history, "RB")


@pytest.fixture
def sandbox(tmp_path, monkeypatch, historical_pairs):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    history = trial.ROOT / "data/graves_history.csv"
    rows = [row for row in trial.read_csv(history) if row["date"] <= "2026-11-04"]
    with (data / "graves_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = {"RB_pt_intercept": 0.02, "RB_pt_slope": 0.67}
    prior = trial.read_csv(data / "graves_history.csv")[-1]
    assert prior["date"] == "2026-11-04"
    baseline = float(prior["nymex_rb"])
    row = {
        "timestamp": at().isoformat(), "commodity": "RB", "prediction_source": "live",
        "predicted_direction": "HIKE", "nymex_move_cents": "2.00",
        "signal_price_used": baseline + 0.02, "baseline_price_used": baseline,
        "signal_contract": "/RBZ26", "baseline_contract": "/RBZ26",
        "contract_provenance_status": "verified", "conviction_provenance": "passthrough_model_v2",
        "settlement_source": "schwab", "baseline_source": "settlement_provenance_verified",
        "settlement_captured_at": at(hour=13, minute=31).isoformat(),
        "conviction_label": "Moderate confidence (82%) | p=0.8200",
        "runtime_config_hash": digest_config(cfg), "actual_next_day_move_cents": "PENDING",
    }
    write_predictions(data, [row])
    trial.register_plan(data, at("2026-11-04", 22, 0))
    # Other tests exercise the historical alignment machinery. Here the frozen
    # historical pairs isolate prospective orchestration from repeated fitting.
    monkeypatch.setattr(alignment, "assert_calibration_alignment", lambda frame: None)
    monkeypatch.setattr(backtest, "training_pairs", lambda frame, prefix: historical_pairs.copy())
    return data, cfg, row


def digest_config(cfg):
    return hashlib.sha256(trial.canonical(cfg).encode()).hexdigest()


def write_predictions(data, rows):
    with (data / "prediction_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_rack(data, day="2026-11-05", price=2.85):
    with (data / "graves_history.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow([day, 2.75, 4.2, price, price + 0.8, 4.4])


def replace_record(path, **updates):
    record = trial.load_record(path)
    record.pop("record_hash")
    record.update(updates)
    path.write_text(json.dumps(trial.seal(record)))


def test_registered_plan_cannot_be_replaced(sandbox):
    data, _, _ = sandbox
    before = (data / "shadow_trial_plan.json").read_bytes()
    first = trial.load_plan(data)
    second = trial.register_plan(data, at("2026-11-04", 23, 0))
    assert first["record_hash"] == second["record_hash"]
    assert (data / "shadow_trial_plan.json").read_bytes() == before
    assert first["primary"]["formal_review_pairs"] == 320
    assert first["automatic_promotion"] is False


def test_late_registration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="before its first prospective verdict"):
        trial.register_plan(tmp_path, at("2026-11-05", 14, 35))
    assert not (tmp_path / "shadow_trial_plan.json").exists()


@pytest.mark.parametrize("moment", [at("2026-11-04"), at(hour=14, minute=34),
                                  at(hour=16, minute=0), at("2026-11-07")])
def test_capture_rejects_pretrial_out_of_window_and_weekends(sandbox, moment):
    data, cfg, _ = sandbox
    assert trial.capture(data, moment, cfg) is None
    assert not (data / "shadow").exists()


def test_capture_seals_prior_information_and_is_idempotent(sandbox):
    data, cfg, _ = sandbox
    before_live = (data / "prediction_log.csv").read_bytes()
    before_history = (data / "graves_history.csv").read_bytes()
    record = trial.capture(data, at(), cfg)
    assert record["status"] == "eligible"
    assert record["production_point_cents"] == pytest.approx(1.36)
    assert len(record["training_pairs"]) == 482
    assert record["training_pairs"][-1][0] < "2026-11-05"
    assert "actual_next_day_move_cents" not in record["production_record"]
    same = trial.capture(data, at(hour=15, minute=10), cfg)
    assert same["record_hash"] == record["record_hash"]
    assert (data / "prediction_log.csv").read_bytes() == before_live
    assert (data / "graves_history.csv").read_bytes() == before_history


@pytest.mark.parametrize("key,value", [
    ("signal_contract", "/RBF27"), ("settlement_source", "live_proxy:schwab"),
    ("runtime_config_hash", "f" * 64), ("prediction_source", "backfill"),
    ("timestamp", at(hour=10, minute=0).isoformat()),
])
def test_unqualified_production_inputs_are_explicitly_ineligible(sandbox, key, value):
    data, cfg, row = sandbox
    row[key] = value
    write_predictions(data, [row])
    record = trial.capture(data, at(), cfg)
    assert record["status"] == "ineligible"
    assert record["reason"]
    assert "training_pairs" not in record


def test_current_rack_outcome_cannot_be_imported_as_live(sandbox):
    data, cfg, _ = sandbox
    append_rack(data)
    record = trial.capture(data, at(), cfg)
    assert record["status"] == "ineligible"
    assert "already available" in record["reason"]


def test_exact_previous_session_is_required(sandbox):
    data, cfg, _ = sandbox
    rows = trial.read_csv(data / "graves_history.csv")
    with (data / "graves_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(row for row in rows if row["date"] != "2026-11-04")
    record = trial.capture(data, at(), cfg)
    assert record["status"] == "ineligible"
    assert "previous-session" in record["reason"]


def test_model_matches_the_research_huber_and_earlier_error_calibration(sandbox):
    data, cfg, _ = sandbox
    captured = trial.capture(data, at(), cfg)
    settings = trial.load_plan(data)["settings"]
    actual = shadow_model.forecast(captured["training_pairs"], captured["nymex_move_cents"], settings)
    values = np.asarray([row[1:] for row in captured["training_pairs"]])
    x, y = values.T
    expected_fit = research.fit_checked(
        research.make_pipeline(research.StandardScaler(), research.HuberRegressor(**research.HYPERPARAMETERS["huber"])),
        x[121:481, None], y[121:481])
    expected = float(expected_fit.predict([[captured["nymex_move_cents"]]])[0])
    assert actual["point_prediction_cents"] == pytest.approx(expected, abs=1e-10)
    probabilities = research.residual_distribution(expected, actual["calibration_errors_cents"])
    assert actual["p_up"] == probabilities[0]
    assert actual["p_down"] == probabilities[1]
    assert actual["experimental_interval_80"] == list(probabilities[2:])
    assert actual["fit_end"] == captured["training_pairs"][-2][0]
    assert actual["calibration_end"] == captured["training_pairs"][-2][0]


def test_purged_pair_does_not_affect_challenger(sandbox):
    data, cfg, _ = sandbox
    captured = trial.capture(data, at(), cfg)
    settings = trial.load_plan(data)["settings"]
    pairs = captured["training_pairs"]
    original = shadow_model.forecast(pairs, 2, settings)
    changed = [list(row) for row in pairs]
    changed[-1][2] += 500
    assert shadow_model.forecast(changed, 2, settings) == original


def test_worker_reads_only_sealed_input_after_capture(sandbox):
    data, cfg, _ = sandbox
    captured = trial.capture(data, at(), cfg)
    append_rack(data, price=9.0)
    # Even if a later outcome exists, no estimator can consume that history.
    result = trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    assert result["status"] == "forecasted"
    assert result["input_hash"] == captured["record_hash"]
    before = trial.record_path(data, "forecasts", "2026-11-05").read_bytes()
    again = trial.run_forecast("2026-11-05", data, at(hour=15, minute=40))
    assert again == result
    assert trial.record_path(data, "forecasts", "2026-11-05").read_bytes() == before


def test_worker_cannot_fabricate_a_missing_input(sandbox):
    data, _, _ = sandbox
    with pytest.raises(ValueError, match="No sealed prospective input"):
        trial.run_forecast("2026-11-05", data, at())


def test_worker_starting_or_finishing_after_deadline_is_late(sandbox):
    data, cfg, _ = sandbox
    trial.capture(data, at(), cfg)
    result = trial.run_forecast("2026-11-05", data, at(hour=15, minute=59),
                                completion_clock=lambda: at(hour=16, minute=0))
    assert result["status"] == "missed_deadline"
    assert "forecast" not in result


def test_model_failure_is_retained_in_itt_scoring(sandbox, monkeypatch):
    data, cfg, _ = sandbox
    captured = trial.capture(data, at(), cfg)
    monkeypatch.setattr(shadow_model, "forecast", Mock(side_effect=RuntimeError("model unavailable")))
    failed = trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    assert failed["status"] == "failed"
    append_rack(data)
    assert trial.reconcile(data, at(hour=20, minute=30)) == 1
    scored = trial.load_record(trial.record_path(data, "outcomes", "2026-11-05"))
    assert scored["on_time_shadow"] is False
    assert scored["shadow_point_cents"] == captured["production_point_cents"]
    assert scored["shadow_itt_abs_error_cents"] == scored["production_abs_error_cents"]
    assert trial.status(data, at(hour=21, minute=0))["failed_or_late_forecasts"] == 1


def test_delayed_email_scores_last_csv_row_and_keeps_forecast_immutable(sandbox):
    data, cfg, _ = sandbox
    trial.capture(data, at(), cfg)
    trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    path = trial.record_path(data, "forecasts", "2026-11-05")
    frozen = path.read_bytes()
    assert trial.reconcile(data, at(hour=21, minute=0)) == 0
    append_rack(data, price=2.9)
    assert trial.reconcile(data, at("2026-11-06", 8, 39)) == 1
    record = trial.load_record(trial.record_path(data, "outcomes", "2026-11-05"))
    prior = trial.load_record(trial.record_path(data, "inputs", "2026-11-05"))["prior_rack_price"]
    assert record["actual_move_cents"] == pytest.approx((2.9 - prior) * 100)
    assert path.read_bytes() == frozen
    assert trial.reconcile(data, at("2026-11-06", 9, 0)) == 0


def test_scoring_detects_later_rack_price_changes(sandbox):
    data, cfg, _ = sandbox
    trial.capture(data, at(), cfg)
    trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    append_rack(data)
    trial.reconcile(data, at(hour=21, minute=0))
    original = (data / "graves_history.csv").read_text()
    (data / "graves_history.csv").write_text(original.replace("2026-11-05,2.75,4.2,2.85,", "2026-11-05,2.75,4.2,3.85,"))
    with pytest.raises(ValueError, match="Previously scored"):
        trial.reconcile(data, at("2026-11-06", 9, 0))


def test_hash_and_semantic_tampering_are_both_rejected(sandbox):
    data, cfg, _ = sandbox
    trial.capture(data, at(), cfg)
    trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    append_rack(data)
    trial.reconcile(data, at(hour=21, minute=0))
    path = trial.record_path(data, "outcomes", "2026-11-05")
    replace_record(path, shadow_itt_abs_error_cents=999)
    with pytest.raises(ValueError, match="calculations are inconsistent"):
        trial.status(data, at(hour=21, minute=0))
    assert "results withheld" in trial.report_text(data)
    raw = path.read_text()
    path.write_text(raw.replace('"record_hash": "', '"record_hash": "bad'))
    with pytest.raises(ValueError, match="hash mismatch"):
        trial.load_record(path)


def test_formal_inference_stays_locked_below_fixed_sample(sandbox):
    data, cfg, _ = sandbox
    trial.capture(data, at(), cfg)
    trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    append_rack(data)
    trial.reconcile(data, at(hour=21, minute=0))
    report = trial.status(data, at(hour=21, minute=0))
    assert report["resolved_itt_pairs"] == 1
    assert report["review_due"] is False
    assert report["primary_inference_available"] is False
    with pytest.raises(ValueError, match="locked"):
        trial.formal_review(data, at(hour=21, minute=0))


def test_completed_missing_sessions_are_visible_and_weekends_are_not(sandbox):
    data, _, _ = sandbox
    summary = trial.status(data, at("2026-11-09", 16, 0))
    assert summary["uncaptured_completed_sessions"] == ["2026-11-05", "2026-11-06", "2026-11-09"]


def test_shadow_capture_failure_cannot_change_or_crash_live_verdict(sandbox, monkeypatch, capsys):
    data, cfg, _ = sandbox
    monkeypatch.setattr(main, "DATA_DIR", str(data))
    monkeypatch.setattr(main, "APP_CONFIG", cfg)
    monkeypatch.setattr(trial, "capture", Mock(side_effect=ValueError("bad shadow input")))
    before = (data / "prediction_log.csv").read_bytes()
    assert main.capture_shadow_input(at()) is None
    assert (data / "prediction_log.csv").read_bytes() == before
    assert "production verdict is unaffected" in capsys.readouterr().out


def test_capture_creates_worker_request_without_changing_live_log(sandbox, monkeypatch, tmp_path):
    data, cfg, _ = sandbox
    marker = tmp_path / "request"
    monkeypatch.setattr(main, "DATA_DIR", str(data))
    monkeypatch.setattr(main, "APP_CONFIG", cfg)
    monkeypatch.setenv("SHADOW_REQUEST_FILE", str(marker))
    main.capture_shadow_input(at())
    assert marker.read_text() == "2026-11-05"
    assert "shadow only" in trial.report_text(data)
    assert "production alerts unchanged" in weekly_report.shadow_trial_report_html(data)


def test_no_publishing_outside_actions(sandbox, monkeypatch):
    data, _, _ = sandbox
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(ValueError, match="restricted"):
        trial.persist(data)


def test_path_traversal_and_implementation_changes_are_rejected(sandbox, monkeypatch):
    data, _, _ = sandbox
    with pytest.raises(ValueError):
        trial.record_path(data, "inputs", "../prediction_log.csv")
    original = trial.file_hash
    monkeypatch.setattr(trial, "file_hash", lambda path: "0" * 64 if Path(path).name == "shadow_model.py" else original(path))
    with pytest.raises(ValueError, match="implementation changed"):
        trial.load_plan(data)


def test_fixed_review_cannot_skip_an_earlier_pending_member(sandbox):
    data, cfg, _ = sandbox
    plan_path = data / "shadow_trial_plan.json"
    plan = trial.load_plan(data)
    # A tiny sample is used only to exercise the review boundary in the fixture.
    primary = {**plan["primary"], "formal_review_pairs": 2, "bootstrap_resamples": 100}
    replace_record(plan_path, primary=primary)
    first = trial.capture(data, at(), cfg)
    trial.run_forecast("2026-11-05", data, at(hour=14, minute=38))
    append_rack(data)
    trial.reconcile(data, at(hour=21, minute=0))
    first_path = trial.record_path(data, "outcomes", "2026-11-05")
    saved = first_path.read_bytes()
    for session, prior in [("2026-11-06", "2026-11-05"), ("2026-11-09", "2026-11-06")]:
        payload = {key: value for key, value in first.items() if key != "record_hash"}
        payload.update(session=session, captured_at=at(session).isoformat(),
                       prior_session=prior, prior_rack_price=2.85,
                       history_latest_session=prior)
        trial.write_once(trial.record_path(data, "inputs", session), payload)
        trial.run_forecast(session, data, at(session, 14, 38))
        append_rack(data, session)
        trial.reconcile(data, at(session, 21, 0))
    first_path.unlink()
    current = trial.status(data, at("2026-11-09", 21, 0))
    assert current["resolved_itt_pairs"] == 2
    assert current["review_due"] is False
    with pytest.raises(ValueError, match="fixed cohort"):
        trial.formal_review(data, at("2026-11-09", 21, 0))
    first_path.write_bytes(saved)
    assert trial.status(data, at("2026-11-09", 21, 0))["review_due"] is True
    review = trial.formal_review(data, at("2026-11-09", 21, 0))
    assert review["n"] == 2
    assert review["start_session"] == "2026-11-05"
    assert review["end_session"] == "2026-11-06"
    assert review["promotion_authorized"] is False
    assert trial.formal_review(data, at("2026-11-10", 21, 0)) == review
