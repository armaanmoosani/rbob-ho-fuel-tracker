"""Isolated prospective gasoline shadow trial and append-only daily records.

Capture/scoring/status use the standard library. ML imports occur only in the
separate worker; none of its forecasts can replace a live purchasing alert.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TZ = ZoneInfo("America/Chicago")
SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instant(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Shadow timestamps require a timezone")
    return parsed.astimezone(TZ)


def seal(payload):
    payload = dict(payload)
    payload["record_hash"] = digest(payload)
    return payload


def load_record(path):
    record = json.loads(Path(path).read_text())
    expected = digest({k: v for k, v in record.items() if k != "record_hash"})
    if record.get("record_hash") != expected:
        raise ValueError(f"Shadow content hash mismatch: {path}")
    return record


def write_once(path, payload):
    """Exclusive creation prevents a retry from replacing a historical record."""
    path = Path(path)
    if path.exists():
        return load_record(path), False
    record = seal(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".shadow-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        return load_record(path), False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record, True


def model_settings():
    return {
        "model": "huber_gasoline_v1", "sklearn_version": "1.7.1",
        "window": 360, "calibration_window": 120, "purge": 1,
        "huber_parameters": {"epsilon": 1.35, "alpha": 0.0001,
                             "max_iter": 1000, "tol": 1e-6},
        "noise_floor_cents": 0.6, "full_confidence": 0.75, "lean_confidence": 0.65,
    }


def register_plan(data_dir=DATA, now=None):
    now = instant((now or datetime.now(timezone.utc)).isoformat())
    path = Path(data_dir) / "shadow_trial_plan.json"
    if path.exists():
        return load_plan(data_dir)
    # Frozen before the first prospective decision, never derived from trial outcomes.
    plan = {
        "schema_version": SCHEMA, "trial_name": "RB Huber prospective v1",
        "registered_at": now.isoformat(), "start_session": "2026-11-05",
        "commodity": "RB", "settings": model_settings(),
        "implementation_hashes": {name: file_hash(ROOT / name) for name in (
            "shadow_trial.py", "shadow_model.py", "alignment.py", "backtest.py",
            "model.py", "futures_util.py", "scratch/compare_ml_models.py",
            "scratch/compare_adaptive_models.py", "requirements-shadow.txt", "requirements.txt")},
        "primary": {"metric": "paired_absolute_error_reduction_cents",
                    "formal_review_pairs": 320, "operational_review_pairs": 60,
                    "alpha_one_sided": 0.025, "planning_power": 0.8,
                    "planning_effect_cents": 0.15,
                    "planning_variance_cents_squared": 0.8759596732890009,
                    "minimum_practical_point_gain_cents": 0.10,
                    "bootstrap_resamples": 20000, "bootstrap_blocks": [5, 10, 20],
                    "optional_stopping": False},
        "minimum_on_time_forecast_coverage": 0.95,
        "failure_policy": "retain_production_forecast_in_intention_to_treat_comparison",
        "automatic_promotion": False,
        "secondary_metrics": ["rmse", "brier_up", "wrong_wait_cost", "gross_timing_proxy"],
        "execution_evidence_required_for_promotion": True,
    }
    start = datetime.combine(date.fromisoformat(plan["start_session"]), datetime.min.time(), TZ).replace(hour=14, minute=35)
    if now >= start:
        raise ValueError("The v1 plan must be registered before its first prospective verdict")
    return write_once(path, plan)[0]


def load_plan(data_dir=DATA, check_implementation=True):
    path = Path(data_dir) / "shadow_trial_plan.json"
    if not path.exists():
        return None
    plan = load_record(path)
    if plan["schema_version"] != SCHEMA or plan["automatic_promotion"] is not False:
        raise ValueError("Invalid shadow trial protocol")
    start = datetime.combine(date.fromisoformat(plan["start_session"]), datetime.min.time(), TZ).replace(hour=14, minute=35)
    if instant(plan["registered_at"]) >= start:
        raise ValueError("Shadow plan was not registered before its start")
    if check_implementation:
        for name, expected in plan["implementation_hashes"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Frozen shadow implementation changed: {name}")
    return plan


def record_path(data_dir, kind, session):
    if date.fromisoformat(session).isoformat() != session:
        raise ValueError("Invalid shadow session date")
    return Path(data_dir) / "shadow" / kind / f"{session}.json"


def numeric(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Shadow numeric inputs must be finite")
    return number


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def capture(data_dir, now, runtime_config):
    """Seal exactly the data available after the normal daily verdict is built."""
    plan = load_plan(data_dir)
    local = instant(now.isoformat())
    if plan is None or local.date().isoformat() < plan["start_session"]:
        return None
    from futures_util import is_nymex_business_day, previous_nymex_business_day
    if not is_nymex_business_day(local.date()) or not (14, 35) <= (local.hour, local.minute) < (16, 0):
        return None
    session = local.date().isoformat()
    path = record_path(data_dir, "inputs", session)
    if path.exists():
        record = load_record(path)
        if record["plan_hash"] != plan["record_hash"]:
            raise ValueError("Existing shadow input belongs to another trial")
        return record
    base = {
        "schema_version": SCHEMA, "plan_hash": plan["record_hash"],
        "session": session, "commodity": "RB", "captured_at": local.isoformat(),
        "status": "ineligible", "reason": "", "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }
    rows = read_csv(Path(data_dir) / "prediction_log.csv")
    matches = [row for row in rows if row.get("prediction_source") == "live"
               and row.get("commodity") == "RB" and row["timestamp"][:10] == session]
    if len(matches) != 1:
        base["reason"] = "No unique live production RB prediction"
        return write_once(path, base)[0]
    row = matches[0]
    # Never carry the later-filled actual outcome into the frozen feature input.
    production = {key: row[key] for key in row if key != "actual_next_day_move_cents"}
    base["production_record"] = production
    prediction_time = instant(row["timestamp"])
    snapshot_time = instant(row["settlement_captured_at"]) if row.get("settlement_captured_at") else None
    valid = (
        row.get("contract_provenance_status") == "verified"
        and row.get("conviction_provenance") == "passthrough_model_v2"
        and row.get("baseline_source") in {"settlement_provenance_verified", "graves_history_same_contract"}
        and row.get("settlement_source") == "schwab"
        and re.fullmatch(r"/RB[FGHJKMNQUVXZ]\d{2}", row.get("signal_contract", ""))
        and row.get("signal_contract") == row.get("baseline_contract")
        and (14, 35) <= (prediction_time.hour, prediction_time.minute) < (16, 0)
        and snapshot_time is not None and snapshot_time.date() == local.date()
        and (13, 30) <= (snapshot_time.hour, snapshot_time.minute) <= (13, 45)
        and snapshot_time <= prediction_time <= local
    )
    if not valid:
        base["reason"] = "Production signal is not an eligible same-contract settlement verdict"
        return write_once(path, base)[0]
    history_path = Path(data_dir) / "graves_history.csv"
    history = read_csv(history_path)
    # A current/future rack row already on disk makes this retrospective.
    if any(item["date"] >= session for item in history):
        base["reason"] = "Current or future rack outcome already available at capture"
        return write_once(path, base)[0]
    previous = previous_nymex_business_day(local.date()).isoformat()
    prior = [item for item in history if item["date"] == previous]
    if len(prior) != 1:
        base["reason"] = "Exact previous-session rack posting is unavailable"
        return write_once(path, base)[0]
    prior_rack = numeric(prior[0]["rack_u"])
    baseline_price = numeric(row["baseline_price_used"])
    signal_price = numeric(row["signal_price_used"])
    if abs(numeric(prior[0]["nymex_rb"]) - baseline_price) > 0.000051:
        base["reason"] = "Recorded baseline differs from the frozen history"
        return write_once(path, base)[0]
    import alignment
    import backtest
    historical_frame = alignment.load_history(str(history_path))
    alignment.assert_calibration_alignment(historical_frame)
    pairs = backtest.training_pairs(alignment.calibration_history(historical_frame), "RB")
    settings = plan["settings"]
    needed = settings["window"] + settings["calibration_window"] + 2 * settings["purge"]
    if len(pairs) < needed:
        base["reason"] = "Insufficient earlier eligible data for frozen Huber calibration"
        return write_once(path, base)[0]
    pairs = pairs.tail(needed)
    if pairs["date"].max().date() >= local.date():
        raise ValueError("Current outcome entered shadow training")
    move = (signal_price - baseline_price) * 100
    if abs(move - numeric(row["nymex_move_cents"])) > 0.011:
        raise ValueError("Production prices and futures move do not agree")
    intercept = numeric(runtime_config["RB_pt_intercept"])
    slope = numeric(runtime_config["RB_pt_slope"])
    # A retry must compare against the exact production calibration captured in
    # the prediction, not a subsequently activated config/cache overlay.
    runtime_hash = hashlib.sha256(canonical(runtime_config).encode()).hexdigest()
    if runtime_hash != row["runtime_config_hash"]:
        base["reason"] = "Production runtime config changed since its logged decision"
        return write_once(path, base)[0]
    base.update({
        "status": "eligible", "reason": "", "prior_session": previous,
        "prior_rack_price": prior_rack, "nymex_move_cents": move,
        "production_point_cents": intercept + slope * move,
        "production_direction": {"HIKE": 1, "DROP": -1, "FLAT": 0}[row["predicted_direction"]],
        "production_probability_correct": numeric(row["conviction_label"].rsplit("| p=", 1)[1]),
        "production_parameters": {"intercept_cents": intercept, "slope": slope},
        "history_sha256": file_hash(history_path),
        "history_latest_session": max(item["date"] for item in history),
        "training_pairs": [[item.date.date().isoformat(), float(item.delta_nymex), float(item.delta_rack)]
                           for item in pairs.itertuples()],
    })
    base["production_p_up"] = (base["production_probability_correct"] if move >= 0
                               else 1 - base["production_probability_correct"])
    return write_once(path, base)[0]


def run_forecast(session, data_dir=DATA, now=None, completion_clock=None):
    plan = load_plan(data_dir)
    if plan is None:
        raise ValueError("Shadow plan is not registered")
    moment = instant((now or datetime.now(timezone.utc)).isoformat())
    source = record_path(data_dir, "inputs", session)
    if not source.exists():
        raise ValueError("No sealed prospective input exists for this session")
    captured = load_record(source)
    path = record_path(data_dir, "forecasts", session)
    if path.exists():
        existing = load_record(path)
        if existing["input_hash"] != captured["record_hash"]:
            raise ValueError("Shadow forecast/input mismatch")
        return existing
    if captured["plan_hash"] != plan["record_hash"]:
        raise ValueError("Shadow input/plan mismatch")
    payload = {
        "schema_version": SCHEMA, "plan_hash": plan["record_hash"],
        "session": session, "input_hash": captured["record_hash"],
        "computed_at": moment.isoformat(), "status": "ineligible", "reason": captured["reason"],
    }
    if captured["status"] == "eligible":
        cutoff = datetime.combine(date.fromisoformat(session), datetime.min.time(), TZ).replace(hour=16)
        if moment < instant(captured["captured_at"]):
            raise ValueError("Shadow forecast time precedes input capture")
        if moment >= cutoff:
            payload.update(status="missed_deadline", reason="Worker missed the frozen 16:00 CT cutoff")
        else:
            try:
                from shadow_model import forecast
                result = forecast(captured["training_pairs"], captured["nymex_move_cents"], plan["settings"])
                if result["fit_end"] >= session or result["calibration_end"] >= session:
                    raise ValueError("Shadow fit/calibration reaches the decision outcome")
                finished = instant((completion_clock() if completion_clock else
                                    (now or datetime.now(timezone.utc))).isoformat())
                payload["computed_at"] = finished.isoformat()
                if finished >= cutoff:
                    payload.update(status="missed_deadline", reason="Model completed after 16:00 CT",
                                   late_forecast=result)
                else:
                    payload.update(status="forecasted", reason="", forecast=result)
            except Exception as exc:
                # A failed challenger remains a recorded failure. It never
                # changes the live verdict or disappears from trial coverage.
                payload.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return write_once(path, payload)[0]


def reconcile(data_dir=DATA, now=None):
    """Append outcomes for sealed decisions, including late emails and last CSV row."""
    plan = load_plan(data_dir)
    if plan is None:
        return 0
    moment = instant((now or datetime.now(timezone.utc)).isoformat())
    history_path = Path(data_dir) / "graves_history.csv"
    history = read_csv(history_path)
    dates = [row["date"] for row in history]
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate history dates prevent shadow outcome attribution")
    lookup = {row["date"]: row for row in history}
    count = 0
    for path in sorted((Path(data_dir) / "shadow" / "inputs").glob("*.json")):
        source = load_record(path)
        session = source["session"]
        if source["plan_hash"] != plan["record_hash"] or source["status"] != "eligible":
            continue
        if instant(source["captured_at"]) > moment or session > moment.date().isoformat() or session not in lookup:
            continue
        prior = lookup.get(source["prior_session"])
        if prior is None or abs(numeric(prior["rack_u"]) - source["prior_rack_price"]) > 0.0000001:
            raise ValueError("Frozen prior-session rack posting changed or is missing")
        outcome_path = record_path(data_dir, "outcomes", session)
        if outcome_path.exists():
            existing = load_record(outcome_path)
            actual = (numeric(lookup[session]["rack_u"]) - source["prior_rack_price"]) * 100
            if existing["input_hash"] != source["record_hash"] or abs(existing["actual_move_cents"] - actual) > 1e-7:
                raise ValueError("Previously scored shadow outcome or input has changed")
            continue
        forecast_path = record_path(data_dir, "forecasts", session)
        if not forecast_path.exists():
            # Record the missing worker as a deadline failure before scoring.
            cutoff = datetime.combine(date.fromisoformat(session), datetime.min.time(), TZ).replace(hour=16)
            if moment < cutoff:
                continue
            run_forecast(session, data_dir, moment)
        predicted = load_record(forecast_path)
        if predicted["input_hash"] != source["record_hash"] or predicted["plan_hash"] != plan["record_hash"]:
            raise ValueError("Cannot score mismatched shadow forecast")
        actual = (numeric(lookup[session]["rack_u"]) - source["prior_rack_price"]) * 100
        success = predicted["status"] == "forecasted"
        shadow_point = (predicted["forecast"]["point_prediction_cents"] if success
                        else source["production_point_cents"])
        shadow_direction = (predicted["forecast"]["direction"] if success else source["production_direction"])
        shadow_p_up = predicted["forecast"]["p_up"] if success else source["production_p_up"]
        payload = {
            "schema_version": SCHEMA, "plan_hash": plan["record_hash"], "session": session,
            "input_hash": source["record_hash"], "forecast_hash": predicted["record_hash"],
            "scored_at": moment.isoformat(), "history_sha256": file_hash(history_path),
            "actual_move_cents": actual, "rack_price": numeric(lookup[session]["rack_u"]),
            "production_point_cents": source["production_point_cents"],
            "shadow_point_cents": shadow_point, "on_time_shadow": success,
            "production_abs_error_cents": abs(source["production_point_cents"] - actual),
            "shadow_itt_abs_error_cents": abs(shadow_point - actual),
            "production_direction": source["production_direction"], "shadow_direction": shadow_direction,
            "production_gross_timing_cents": source["production_direction"] * actual,
            "shadow_itt_gross_timing_cents": shadow_direction * actual,
            "production_squared_error": (source["production_point_cents"] - actual) ** 2,
            "shadow_itt_squared_error": (shadow_point - actual) ** 2,
            "production_brier_up": (source["production_p_up"] - float(actual > 0)) ** 2,
            "shadow_itt_brier_up": (shadow_p_up - float(actual > 0)) ** 2,
            "production_wrong_wait_cost_cents": max(actual, 0) if source["production_direction"] == -1 else 0,
            "shadow_itt_wrong_wait_cost_cents": max(actual, 0) if shadow_direction == -1 else 0,
            "realized_savings_dollars": None,
        }
        count += int(write_once(outcome_path, payload)[1])
    return count


def trial_records(data_dir, kind, plan):
    records = []
    for path in sorted((Path(data_dir) / "shadow" / kind).glob("*.json")):
        row = load_record(path)
        if (row["plan_hash"] != plan["record_hash"] or row["session"] < plan["start_session"]
                or row["session"] != path.stem):
            raise ValueError("Shadow record violates the registered trial boundary")
        records.append(row)
    return records


def audit_links(inputs, forecasts, outcomes, plan):
    """Check referential integrity and recompute scores, not just content hashes."""
    sources = {row["session"]: row for row in inputs}
    predictions = {row["session"]: row for row in forecasts}
    for source in inputs:
        moment = instant(source["captured_at"])
        if (source["schema_version"] != SCHEMA or moment.date().isoformat() != source["session"]
                or not (14, 35) <= (moment.hour, moment.minute) < (16, 0)
                or source["status"] not in {"eligible", "ineligible"}):
            raise ValueError("Invalid prospective input boundary")
        if source["status"] == "eligible":
            dates = [row[0] for row in source["training_pairs"]]
            if dates != sorted(set(dates)) or dates[-1] >= source["session"]:
                raise ValueError("Invalid shadow training cutoff")
            if source["history_latest_session"] >= source["session"]:
                raise ValueError("Shadow input contains a current/future outcome")
    for predicted in forecasts:
        source = sources.get(predicted["session"])
        if source is None or predicted["input_hash"] != source["record_hash"]:
            raise ValueError("Shadow forecast does not match its sealed input")
        computed = instant(predicted["computed_at"])
        if computed < instant(source["captured_at"]):
            raise ValueError("Shadow forecast predates its input")
        state = predicted["status"]
        if state not in {"forecasted", "failed", "missed_deadline", "ineligible"}:
            raise ValueError("Unknown shadow forecast status")
        if (state == "ineligible") != (source["status"] == "ineligible"):
            raise ValueError("Shadow eligibility changed after capture")
        if state == "forecasted":
            fit = predicted["forecast"]
            if computed.date().isoformat() != source["session"] or computed.hour >= 16:
                raise ValueError("Late forecast incorrectly marked on time")
            if fit["fit_end"] >= source["session"] or fit["calibration_end"] >= source["session"]:
                raise ValueError("Shadow forecast leaks its current outcome")
            if (not 0 <= numeric(fit["p_up"]) <= 1 or not 0 <= numeric(fit["p_down"]) <= 1
                    or fit["p_up"] + fit["p_down"] > 1.00000001
                    or fit["direction"] not in (-1, 0, 1)):
                raise ValueError("Invalid shadow probability or action")
    for row in outcomes:
        source = sources.get(row["session"])
        predicted = predictions.get(row["session"])
        if source is None or predicted is None or row["input_hash"] != source["record_hash"] or row["forecast_hash"] != predicted["record_hash"]:
            raise ValueError("Shadow outcome references a different forecast or input")
        if instant(row["scored_at"]) < instant(predicted["computed_at"]):
            raise ValueError("Shadow outcome was scored before its forecast")
        actual = (numeric(row["rack_price"]) - source["prior_rack_price"]) * 100
        success = predicted["status"] == "forecasted"
        point = predicted["forecast"]["point_prediction_cents"] if success else source["production_point_cents"]
        direction = predicted["forecast"]["direction"] if success else source["production_direction"]
        p_up = predicted["forecast"]["p_up"] if success else source["production_p_up"]
        checks = {
            "actual_move_cents": actual, "production_point_cents": source["production_point_cents"],
            "shadow_point_cents": point,
            "production_abs_error_cents": abs(source["production_point_cents"] - actual),
            "shadow_itt_abs_error_cents": abs(point - actual),
            "production_squared_error": (source["production_point_cents"] - actual) ** 2,
            "shadow_itt_squared_error": (point - actual) ** 2,
            "production_brier_up": (source["production_p_up"] - float(actual > 0)) ** 2,
            "shadow_itt_brier_up": (p_up - float(actual > 0)) ** 2,
            "production_gross_timing_cents": source["production_direction"] * actual,
            "shadow_itt_gross_timing_cents": direction * actual,
            "production_wrong_wait_cost_cents": max(actual, 0) if source["production_direction"] == -1 else 0,
            "shadow_itt_wrong_wait_cost_cents": max(actual, 0) if direction == -1 else 0,
        }
        if row["on_time_shadow"] is not success or row["shadow_direction"] != direction:
            raise ValueError("Shadow outcome mislabels a failure or action")
        if any(abs(numeric(row[key]) - value) > 1e-7 for key, value in checks.items()):
            raise ValueError("Shadow outcome calculations are inconsistent")


def status(data_dir=DATA, now=None):
    plan = load_plan(data_dir)
    if plan is None:
        return {"active": False, "message": "Shadow trial not registered"}
    inputs = trial_records(data_dir, "inputs", plan)
    forecasts = trial_records(data_dir, "forecasts", plan)
    outcomes = trial_records(data_dir, "outcomes", plan)
    audit_links(inputs, forecasts, outcomes, plan)
    n = len(outcomes)
    eligible = sum(row["status"] == "eligible" for row in inputs)
    success = sum(row["status"] == "forecasted" for row in forecasts)
    target = plan["primary"]["formal_review_pairs"]
    cohort = [row for row in inputs if row["status"] == "eligible"][:target]
    resolved_dates = {row["session"] for row in outcomes}
    complete_cohort = len(cohort) == target and all(row["session"] in resolved_dates for row in cohort)
    result = {
        "active": True, "trial_name": plan["trial_name"], "plan_hash": plan["record_hash"],
        "start_session": plan["start_session"], "captured_sessions": len(inputs),
        "eligible_inputs": eligible, "ineligible_inputs": len(inputs) - eligible,
        "on_time_forecasts": success,
        "failed_or_late_forecasts": sum(row["status"] in {"failed", "missed_deadline"} for row in forecasts),
        "unprocessed_inputs": len(inputs) - len(forecasts),
        "resolved_itt_pairs": n, "pending_pairs": eligible - n,
        "formal_review_pairs": target,
        "operational_review_due": n >= plan["primary"]["operational_review_pairs"],
        "review_due": complete_cohort and not (Path(data_dir) / "shadow" / "reviews" / "primary.json").exists(),
        "automatic_promotion": False,
        "live_predictions_used_in_production_statistics": 0,
    }
    # Explicitly expose days where the tracker never produced a capture. These
    # cannot be fabricated later or silently converted into live predictions.
    from futures_util import us_market_holidays
    current = instant((now or datetime.now(timezone.utc)).isoformat())
    end = current.date() if current.hour >= 16 else current.date() - timedelta(days=1)
    day = date.fromisoformat(plan["start_session"])
    captured_dates = {row["session"] for row in inputs}
    missing = []
    while day <= end:
        # Match the live tracker's holiday suppression, including holidays
        # where Globex has a partial session but the tracker sends no verdict.
        if day.weekday() < 5 and day not in us_market_holidays(day.year) and day.isoformat() not in captured_dates:
            missing.append(day.isoformat())
        day += timedelta(days=1)
    result["uncaptured_completed_sessions"] = missing
    if outcomes:
        result.update(
            production_mae_cents=statistics.mean(row["production_abs_error_cents"] for row in outcomes),
            shadow_itt_mae_cents=statistics.mean(row["shadow_itt_abs_error_cents"] for row in outcomes),
            primary_inference_available=complete_cohort,
            production_rmse_cents=math.sqrt(statistics.mean(row["production_squared_error"] for row in outcomes)),
            shadow_itt_rmse_cents=math.sqrt(statistics.mean(row["shadow_itt_squared_error"] for row in outcomes)),
            production_brier_up=statistics.mean(row["production_brier_up"] for row in outcomes),
            shadow_itt_brier_up=statistics.mean(row["shadow_itt_brier_up"] for row in outcomes),
        )
    else:
        result["primary_inference_available"] = False
    return result


def formal_review(data_dir=DATA, now=None):
    plan = load_plan(data_dir)
    path = Path(data_dir) / "shadow" / "reviews" / "primary.json"
    if path.exists():
        return load_record(path)
    quality = status(data_dir, now)
    if not quality.get("primary_inference_available"):
        raise ValueError("Formal inference is locked until the fixed cohort is complete")
    resolved = {row["session"]: row for row in trial_records(data_dir, "outcomes", plan)}
    cohort = [row for row in trial_records(data_dir, "inputs", plan) if row["status"] == "eligible"]
    target = plan["primary"]["formal_review_pairs"]
    rows = [resolved[row["session"]] for row in cohort[:target]]
    import numpy as np
    from scratch.compare_ml_models import paired_block_intervals
    gains = np.asarray([row["production_abs_error_cents"] - row["shadow_itt_abs_error_cents"] for row in rows])
    confidence = 1 - 2 * plan["primary"]["alpha_one_sided"]
    bounds = {str(block): paired_block_intervals(
        gains[:, None], block, resamples=plan["primary"]["bootstrap_resamples"],
        confidence=confidence)[:, :].ravel().tolist() for block in plan["primary"]["bootstrap_blocks"]}
    coverage = statistics.mean(row["on_time_shadow"] for row in rows)
    cohort_end = datetime.combine(date.fromisoformat(rows[-1]["session"]), datetime.min.time(), TZ).replace(hour=16)
    quality = status(data_dir, cohort_end)
    return write_once(path, {
        "schema_version": SCHEMA, "plan_hash": plan["record_hash"],
        "session": rows[-1]["session"], "reviewed_at": (now or datetime.now(timezone.utc)).isoformat(),
        "n": target, "start_session": rows[0]["session"], "end_session": rows[-1]["session"],
        "outcome_hashes": [row["record_hash"] for row in rows],
        "mean_itt_mae_gain_cents": float(gains.mean()), "approximate_block_intervals": bounds,
        "positive_lower_bound_all_blocks": all(value[0] > 0 for value in bounds.values()),
        "on_time_forecast_coverage": coverage,
        "uncaptured_completed_sessions": quality["uncaptured_completed_sessions"],
        "meets_forecast_effect_and_coverage_gates": bool(
            all(value[0] > 0 for value in bounds.values())
            and float(gains.mean()) >= plan["primary"]["minimum_practical_point_gain_cents"]
            and coverage >= plan["minimum_on_time_forecast_coverage"]
            and not quality["uncaptured_completed_sessions"]),
        "promotion_authorized": False,
        "note": "Forecast evidence only. Risk and actual execution evidence require separate review.",
    })[0]


def report_text(data_dir=DATA):
    try:
        summary = status(data_dir)
    except Exception as exc:
        return f"Shadow trial integrity check failed; results withheld: {exc}"
    if not summary["active"]:
        return summary["message"]
    text = (f"Gasoline Huber shadow only; production alerts unchanged. Starts {summary['start_session']}. "
            f"{summary['resolved_itt_pairs']}/{summary['formal_review_pairs']} resolved fixed-review pairs; "
            f"{summary['on_time_forecasts']} on-time forecasts, {summary['failed_or_late_forecasts']} failed/late, "
            f"{summary['pending_pairs']} pending. {len(summary['uncaptured_completed_sessions'])} completed "
            "sessions lack a capture. No automatic promotion or realized-savings claim.")
    if "production_mae_cents" in summary:
        text += (f" Descriptive MAE: production {summary['production_mae_cents']:.3f} cents, "
                 f"shadow including fallback {summary['shadow_itt_mae_cents']:.3f} cents.")
    if not summary["primary_inference_available"]:
        text += " Formal inference remains locked; early comparisons are descriptive."
    return text


def persist(data_dir=DATA):
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise ValueError("Shadow publishing is restricted to GitHub Actions")
    relative = str((Path(data_dir) / "shadow").relative_to(ROOT))
    if not (Path(data_dir) / "shadow").exists():
        print("No shadow records to persist.")
        return
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True, cwd=ROOT)
    subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=True, cwd=ROOT)
    subprocess.run(["git", "add", relative], check=True, cwd=ROOT)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("No new shadow records.")
        return
    subprocess.run(["git", "commit", "-m", "Record isolated Huber shadow trial"], check=True, cwd=ROOT)
    for _ in range(5):
        pulled = subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT).returncode == 0
        if pulled and subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode == 0:
            return
        subprocess.run(["git", "rebase", "--abort"], cwd=ROOT, capture_output=True)
    raise RuntimeError("Could not persist shadow records after five attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "forecast", "reconcile", "status", "review", "persist", "verify-model"])
    parser.add_argument("--date")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()
    if args.command == "register":
        result = register_plan()
    elif args.command == "forecast":
        if not args.date:
            parser.error("forecast requires --date and a sealed prospective input")
        result = run_forecast(args.date)
    elif args.command == "reconcile":
        result = {"new_outcomes": reconcile()}
    elif args.command == "review":
        result = formal_review()
    elif args.command == "persist":
        persist()
        result = {"persisted": True}
    elif args.command == "verify-model":
        plan = load_plan()
        import sklearn
        if sklearn.__version__ != plan["settings"]["sklearn_version"]:
            raise ValueError("Frozen shadow sklearn version mismatch")
        result = {"verified": True, "plan_hash": plan["record_hash"], "sklearn_version": sklearn.__version__}
    else:
        result = status()
    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
            handle.write(f"review_due={str(result.get('review_due', False)).lower()}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
