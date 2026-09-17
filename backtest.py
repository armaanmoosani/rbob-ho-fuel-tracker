"""Nightly calibration of the rack pass-through model.

What changed and why
--------------------
The previous engine fitted a threshold as the ``Hp``-th percentile of NYMEX
moves on days that happened to be hikes, then grid-searched ``(W, Hp, Dp)`` over
64 combinations to maximise the median savings of three walk-forward folds.
Three things were wrong with that:

1. It trained on the whole of ``graves_history.csv``, two thirds of which is
   stamped one session late (see ``alignment.py``).  Roughly 60% of the
   training pairs matched a rack price against the wrong settle.
2. It selected hyper-parameters by maximising the same folds it then reported
   as out-of-sample.  The objective surface is in fact almost flat -- all 64
   combinations landed within 2.7% of the best -- so the search was selecting
   noise while lending the result unearned credibility.
3. A percentile threshold carries no probabilistic meaning, so the system could
   not state how likely any individual signal was to be right.

The engine now fits one interpretable model (``model.PassThrough``) on
alignment-verified rows only, places thresholds where the win probability
reaches a configured target, and floors them at the measured live snapshot
error.  The walk-forward is retained purely as an *evaluation*: nothing is
selected from it, so its numbers are an honest out-of-sample estimate.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import pytz

import alignment
import model
import validate_data
from calibration_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    append_calibration_artifact,
    artifact_id,
    load_calibration_artifacts,
)
from futures_util import is_contract_roll_day, is_nymex_business_day

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
METRICS_CACHE_PATH = os.path.join(DATA_DIR, "metrics_cache.json")
CALIBRATION_RUNS_PATH = os.path.join(DATA_DIR, "calibration_runs.jsonl")
CALIBRATION_PURGE_ROWS = 1
CALIBRATION_METHOD_VERSION = "v2-passthrough-confidence"

# Evaluation-only walk-forward geometry.  No parameter is chosen from these
# folds.  Widened after the re-dating migration tripled the usable history:
# six 45-session blocks span about 16 months, so the reported figure now covers
# a calm regime as well as a volatile one instead of only the recent spike.
EVAL_TEST_ROWS = 45
EVAL_FOLDS = 6


DEFAULTS = {
    "MIN_ROWS_FOR_TUNING": 120,
    "PRICE_MIN": 1.50,
    "PRICE_MAX": 6.00,
    # Confidence the marginal signal must reach before it is an alert.  This is
    # a policy choice about how selective to be, not a fitted parameter.
    "TARGET_SIGNAL_CONFIDENCE": 0.75,
    "TARGET_LEAN_CONFIDENCE": 0.65,
    # Floor applied when the live baseline comes from the same source the model
    # was fitted on.  Decomposing 74 non-roll live decisions showed the 1:30 PM
    # signal price matches the recorded settle to 0.000c, so with a
    # calibration-matched baseline the delta carries no measurement error at
    # all.  0.6c is kept as insurance against an unnoticed source drift, and
    # captures essentially all of the available alerts: HO gains 28 alerts going
    # from 1.2 to 0.6, and only one more going from 0.6 to 0.3.
    "SNAPSHOT_NOISE_FLOOR_CENTS": 0.6,
    # Floor applied at decision time when the baseline is NOT the calibration
    # source -- the old universal value, sized to the p95 of that mismatch.
    "FALLBACK_NOISE_FLOOR_CENTS": 1.2,
    # Rows used for the final fit.  Raised from 180 after the re-dating
    # migration: measured on an identical evaluation block, lengthening the
    # window is the one change that improved calibration out of sample.
    #   W     RB gap   RB skill    HO gap   HO skill
    #   120   +0.086    -0.012     +0.039    +0.068
    #   180   +0.081    -0.027     +0.027    +0.118
    #   240   +0.061    +0.031     +0.017    +0.149
    #   360   +0.049    +0.063     +0.008    +0.143
    #   500   +0.042    +0.077     +0.000    +0.140
    # 360 takes almost all of the gain while leaving room for the folds.
    "ROLLING_WINDOW_DAYS": 360,
    "LAG_DAYS": 0,
}


def load_config():
    cfg = DEFAULTS.copy()
    for path in (CONFIG_PATH, METRICS_CACHE_PATH):
        if os.path.exists(path):
            try:
                with open(path, "r") as handle:
                    cfg.update(json.load(handle))
            except Exception:
                pass
    return cfg


# Keys written by the superseded percentile engine.  They must be dropped
# rather than left to age in the cache: several were in-sample statistics
# (``*_historical_win_rate`` was 0.94 measured on its own training window) and
# nothing reads them any more, so a future reader would take them for live
# performance figures.
SUPERSEDED_CACHE_KEY_SUFFIXES = (
    "_historical_win_rate", "_average_savings", "_wait_upside_cvar_95",
    "_high_z_win_rate", "_high_z_savings", "_high_z_count",
    "_mod_z_win_rate", "_mod_z_savings", "_mod_z_count",
    "_low_z_win_rate", "_low_z_savings", "_low_z_count",
    "_opt_Hp", "_opt_Dp",
)


def _is_superseded(key):
    return any(key.endswith(suffix) for suffix in SUPERSEDED_CACHE_KEY_SUFFIXES)


def save_metrics_cache(cfg, effective_session=None, source_history_hash=None):
    output_keys = ["ROLLING_WINDOW_DAYS", "LAG_DAYS",
                   "TARGET_SIGNAL_CONFIDENCE", "TARGET_LEAN_CONFIDENCE",
                   "SNAPSHOT_NOISE_FLOOR_CENTS", "FALLBACK_NOISE_FLOOR_CENTS",
                   "CALIBRATION_ERA_START"]
    output_keys.extend(k for k in cfg if k.startswith(("RB_", "HO_")))
    cache_data = {k: cfg[k] for k in dict.fromkeys(output_keys)
                  if k in cfg and not _is_superseded(k)}
    cache_data["CALIBRATION_EFFECTIVE_SESSION"] = (
        str(effective_session)[:10] if effective_session else
        datetime.now(pytz.timezone("America/Chicago")).date().isoformat()
    )
    cache_data["CALIBRATION_METHOD_VERSION"] = CALIBRATION_METHOD_VERSION
    if source_history_hash:
        cache_data["CALIBRATION_SOURCE_HISTORY_HASH"] = source_history_hash

    tmp_path = METRICS_CACHE_PATH + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(cache_data, handle, indent=2)
    os.replace(tmp_path, METRICS_CACHE_PATH)


def _push_is_authorised():
    """Only the scheduled workflow may publish, unless asked explicitly.

    ``main()`` used to commit and push unconditionally, so merely running
    ``python3 backtest.py`` locally -- to inspect a calibration, or from a test
    harness -- published to the production branch. That is a destructive
    default for a script whose main job is to compute numbers.

    Publishing now requires either GitHub Actions (``GITHUB_ACTIONS=true``) or
    an explicit ``--commit`` / ``BACKTEST_ALLOW_PUSH=1``.
    """
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return True
    if os.environ.get("BACKTEST_ALLOW_PUSH") == "1":
        return True
    return "--commit" in sys.argv


def git_commit_push(message):
    if not _push_is_authorised():
        print(f"Calibration written locally; not committing.\n"
              f"  Would have committed: {message}\n"
              f"  Pass --commit (or set BACKTEST_ALLOW_PUSH=1) to publish.")
        return
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "--global", "user.email",
                        "github-actions[bot]@users.noreply.github.com"], check=True)
        paths = ["data/metrics_cache.json", "data/integrity_hashes.csv"]
        if os.path.exists(CALIBRATION_RUNS_PATH):
            paths.append("data/calibration_runs.jsonl")
        subprocess.run(["git", "add", *paths], check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if staged.returncode not in (0, 1):
            raise subprocess.CalledProcessError(staged.returncode, staged.args)
        if staged.returncode != 0:
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
            print("Successfully committed and pushed metrics changes.")
        else:
            print("No metrics changes to commit.")
    except Exception as exc:
        print(f"Git commit/push failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Training pairs
# ---------------------------------------------------------------------------

def training_pairs(df, prefix):
    """Alignment-verified, roll-free (nymex_delta, rack_delta) pairs.

    Roll sessions are dropped because the settle difference on them spans two
    different contracts.  With the roll detector repaired those sessions carry a
    mean residual of about +4 cents against about 0 elsewhere, so leaving them
    in would bias both the fitted intercept and the tail-risk estimate.
    """
    frame = alignment.aligned_deltas(df, prefix)
    if frame.empty:
        return frame
    keep = ~frame["date"].apply(lambda d: is_contract_roll_day(d.date(), prefix))
    return frame[keep].reset_index(drop=True)


def walk_forward_evaluation(frame, cfg, test_rows=EVAL_TEST_ROWS, folds=EVAL_FOLDS,
                            purge_rows=CALIBRATION_PURGE_ROWS):
    """Honest out-of-sample estimate.  Nothing is selected from these folds.

    Each fold fits on the rows before a purge gap and scores the block after it.
    Because no parameter is chosen by comparing folds, the aggregate is a
    genuine out-of-sample figure rather than the maximum of a search.
    """
    window = int(cfg["ROLLING_WINDOW_DAYS"])
    target = float(cfg["TARGET_SIGNAL_CONFIDENCE"])
    floor = float(cfg["SNAPSHOT_NOISE_FLOOR_CENTS"])
    n = len(frame)

    needed = folds * test_rows + purge_rows + model.MIN_FIT_ROWS
    if n < needed:
        raise model.ModelFitError(
            f"{n} usable pairs but the walk-forward needs {needed} "
            f"({folds} folds x {test_rows} test rows + {purge_rows} purge + "
            f"{model.MIN_FIT_ROWS} minimum fit rows)"
        )

    results = []
    for fold in range(folds):
        test_end = n - fold * test_rows
        test_start = test_end - test_rows
        train_end = test_start - purge_rows
        train_start = max(0, train_end - window)
        if train_end - train_start < model.MIN_FIT_ROWS:
            continue
        train = frame.iloc[train_start:train_end]
        test = frame.iloc[test_start:test_end]
        fit = model.fit_passthrough(train["delta_nymex"], train["delta_rack"])
        hike, drop = model.apply_noise_floor(
            *fit.threshold_for_confidence(target), floor)
        stats_ = model.summarize(test["delta_nymex"], test["delta_rack"], hike, drop)
        stats_.pop("payoff", None)
        stats_.update({
            "fold": fold,
            "test_start": test["date"].iloc[0].date().isoformat(),
            "test_end": test["date"].iloc[-1].date().isoformat(),
            "hike": hike,
            "drop": drop,
        })
        results.append(stats_)

    if not results:
        raise model.ModelFitError("no walk-forward fold had enough training rows")

    alerts = sum(r["alerts"] for r in results)
    correct = sum(r["correct"] for r in results)
    return {
        "folds": list(reversed(results)),
        "alerts": alerts,
        "correct": correct,
        "precision": (correct / alerts) if alerts else float("nan"),
        "total_savings": float(sum(r["total_savings"] for r in results)),
        "mean_savings": (float(sum(r["total_savings"] for r in results)) / alerts)
                        if alerts else float("nan"),
    }


def calibrate(df, prefix, cfg):
    """Fit one commodity and write its results into ``cfg``.

    Raises rather than falling back.  The previous engine returned a sentinel
    ``-9999`` when no fold could be built and then silently reverted to a
    hardcoded ``(120, 15, 85)``, so a configuration that did not fit the data
    produced plausible-looking thresholds with no warning anywhere.
    """
    frame = training_pairs(df, prefix)
    window = int(cfg["ROLLING_WINDOW_DAYS"])
    target = float(cfg["TARGET_SIGNAL_CONFIDENCE"])
    lean_target = float(cfg["TARGET_LEAN_CONFIDENCE"])
    floor = float(cfg["SNAPSHOT_NOISE_FLOOR_CENTS"])

    evaluation = walk_forward_evaluation(frame, cfg)

    final = frame.tail(window)
    fit = model.fit_passthrough(final["delta_nymex"], final["delta_rack"])
    hike, drop = model.apply_noise_floor(*fit.threshold_for_confidence(target), floor)
    lean_hike, lean_drop = model.apply_noise_floor(
        *fit.threshold_for_confidence(lean_target), floor)
    # A lean band only exists where it is strictly weaker than a full alert.
    # After the noise floor is applied the two can coincide, in which case the
    # honest thing is to have no lean band rather than a duplicate alert.
    lean_hike = min(lean_hike, hike)
    lean_drop = max(lean_drop, drop)

    in_window = model.summarize(final["delta_nymex"], final["delta_rack"], hike, drop)
    # Tail risk is estimated over the whole alignment-verified era rather than
    # the fit window.  A 180-row window leaves under 20 WAIT observations for
    # RB, which cannot support any tail statistic at all, and the shape of the
    # adverse tail changes far more slowly than the threshold does.
    tail = model.wait_tail_risk(frame["delta_nymex"], frame["delta_rack"], drop)

    cfg.update(model.passthrough_to_config(fit, prefix))
    cfg[f"{prefix}_HIKE_THRESHOLD_CENTS"] = round(hike, 2)
    cfg[f"{prefix}_DROP_THRESHOLD_CENTS"] = round(drop, 2)
    cfg[f"{prefix}_LEAN_HIKE_CENTS"] = round(lean_hike, 2)
    cfg[f"{prefix}_LEAN_DROP_CENTS"] = round(lean_drop, 2)
    cfg[f"{prefix}_window_days"] = window
    cfg[f"{prefix}_nymex_daily_std"] = round(float(final["delta_nymex"].std(ddof=1)), 4)

    # Out-of-sample figures.  These are the only performance numbers any
    # user-facing surface is allowed to quote.
    cfg[f"{prefix}_oos_alerts"] = evaluation["alerts"]
    cfg[f"{prefix}_oos_precision"] = round(evaluation["precision"], 4)
    cfg[f"{prefix}_oos_mean_savings"] = round(evaluation["mean_savings"], 4)
    cfg[f"{prefix}_oos_total_savings"] = round(evaluation["total_savings"], 4)
    cfg[f"{prefix}_oos_window"] = (
        f"{evaluation['folds'][0]['test_start']}..{evaluation['folds'][-1]['test_end']}"
    )
    # In-window figures, kept only for diagnostics and explicitly named so they
    # can never be mistaken for out-of-sample performance.
    cfg[f"{prefix}_insample_alerts"] = in_window["alerts"]
    cfg[f"{prefix}_insample_precision"] = round(in_window["precision"], 4)

    for stale in (f"{prefix}_wait_cvar_95", f"{prefix}_wait_cvar_low",
                  f"{prefix}_wait_cvar_high", f"{prefix}_wait_cvar_tail_n",
                  f"{prefix}_wait_cvar_sample_n", f"{prefix}_wait_adverse_probability",
                  f"{prefix}_wait_median_move"):
        cfg.pop(stale, None)
    if tail is None:
        cfg[f"{prefix}_wait_cvar_status"] = "insufficient_tail"
    else:
        cfg[f"{prefix}_wait_cvar_95"] = round(tail["cvar"], 4)
        cfg[f"{prefix}_wait_cvar_low"] = round(tail["cvar_low"], 4)
        cfg[f"{prefix}_wait_cvar_high"] = round(tail["cvar_high"], 4)
        cfg[f"{prefix}_wait_cvar_tail_n"] = tail["tail_n"]
        cfg[f"{prefix}_wait_cvar_sample_n"] = tail["sample_n"]
        cfg[f"{prefix}_wait_adverse_probability"] = round(tail["probability_adverse"], 4)
        cfg[f"{prefix}_wait_median_move"] = round(tail["median_move"], 4)
        cfg[f"{prefix}_wait_cvar_status"] = "ok"

    message = (f"b={fit.slope:.3f} R2={fit.r2:.3f} hike={hike:+.2f}c drop={drop:+.2f}c "
               f"OOS prec={evaluation['precision']:.1%} on {evaluation['alerts']} alerts")
    return cfg, message, evaluation


# ---------------------------------------------------------------------------
# Point-in-time artifact ledger (unchanged machinery, new calibration inside)
# ---------------------------------------------------------------------------

def _next_nymex_business_session(day):
    session = pd.Timestamp(day).date() + pd.Timedelta(days=1)
    while not is_nymex_business_day(session):
        session += pd.Timedelta(days=1)
    return session.isoformat()


def _history_hash(df):
    """Stable hash of exactly the rows eligible to train one artifact."""
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    payload = normalized.to_csv(index=False, lineterminator="\n", float_format="%.10f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _calibration_payload(cfg):
    keys = ["ROLLING_WINDOW_DAYS", "LAG_DAYS", "TARGET_SIGNAL_CONFIDENCE",
            "TARGET_LEAN_CONFIDENCE", "SNAPSHOT_NOISE_FLOOR_CENTS",
            "FALLBACK_NOISE_FLOOR_CENTS", "CALIBRATION_ERA_START"]
    keys.extend(key for key in cfg if key.startswith(("RB_", "HO_")))
    return {key: cfg[key] for key in sorted(set(keys)) if key in cfg}


def _eligible_training_history(df, effective_session=None):
    clean = alignment.calibration_history(alignment.load_history_from_frame(df))
    if len(clean) <= CALIBRATION_PURGE_ROWS:
        raise ValueError("Not enough history to create a purged calibration artifact.")

    requested_session = pd.Timestamp(effective_session).date() if effective_session else None
    matching = (clean.index[clean["date"].dt.date == requested_session].tolist()
                if requested_session else [])
    training_end_index = matching[0] - 1 if matching else len(clean) - 1
    if training_end_index < 0:
        raise ValueError("No completed rack outcome exists before this calibration session.")
    return clean, clean.iloc[:training_end_index + 1].copy()


def calibration_is_current(cfg, source_history_hash, effective_session,
                           latest_history_session):
    """Return whether this exact history has already produced the live cache."""
    cached_hash = cfg.get("CALIBRATION_SOURCE_HISTORY_HASH")
    cached_session = str(cfg.get("CALIBRATION_EFFECTIVE_SESSION", ""))[:10]
    cached_method = cfg.get("CALIBRATION_METHOD_VERSION")
    # A method change must always force a recalibration, otherwise the cache
    # would keep serving thresholds produced by the superseded engine.
    if cached_method != CALIBRATION_METHOD_VERSION:
        return False
    if cached_hash:
        return cached_hash == source_history_hash and cached_session == effective_session
    return cached_session in {str(latest_history_session)[:10], effective_session}


def build_shadow_calibration_artifact(df, cfg, effective_session=None,
                                      prior_artifact=None):
    """Build, but do not persist, the calibration eligible for the next session."""
    clean, training_df = _eligible_training_history(df, effective_session)
    effective_session = effective_session or _next_nymex_business_session(clean["date"].iloc[-1])

    artifact_cfg = dict(cfg)
    if prior_artifact:
        artifact_cfg.update(prior_artifact["calibration"])

    policy = {
        "TARGET_SIGNAL_CONFIDENCE": artifact_cfg.get("TARGET_SIGNAL_CONFIDENCE"),
        "TARGET_LEAN_CONFIDENCE": artifact_cfg.get("TARGET_LEAN_CONFIDENCE"),
        "SNAPSHOT_NOISE_FLOOR_CENTS": artifact_cfg.get("SNAPSHOT_NOISE_FLOOR_CENTS"),
        "ROLLING_WINDOW_DAYS": artifact_cfg.get("ROLLING_WINDOW_DAYS"),
        "CALIBRATION_ERA_START": alignment.CALIBRATION_ERA_START,
    }

    for prefix in ("RB", "HO"):
        artifact_cfg, _, _ = calibrate(training_df, prefix, artifact_cfg)
    artifact_cfg["LAG_DAYS"] = 0

    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "effective_session": str(effective_session)[:10],
        "training_start": training_df["date"].iloc[0].date().isoformat(),
        "training_end": training_df["date"].iloc[-1].date().isoformat(),
        "purge_rows": CALIBRATION_PURGE_ROWS,
        "source_history_hash": _history_hash(training_df),
        "source_row_count": len(training_df),
        "candidate_grid_version": CALIBRATION_METHOD_VERSION,
        "candidate_grid": policy,
        "objective": ("thresholds placed where P(correct) reaches "
                      "TARGET_SIGNAL_CONFIDENCE under the empirical residual "
                      "distribution, floored at the measured snapshot error; "
                      "no parameter is selected from the walk-forward folds"),
        "smoothing_input": policy,
        "prior_artifact_id": prior_artifact["artifact_id"] if prior_artifact else "bootstrap_config",
        "calibration": _calibration_payload(artifact_cfg),
        "generated_at": datetime.now(pytz.timezone("America/Chicago")).isoformat(),
    }
    artifact["artifact_id"] = artifact_id(artifact)
    return artifact


def write_shadow_calibration_artifact(df, cfg):
    """Persist one immutable next-session artifact."""
    artifacts = load_calibration_artifacts(CALIBRATION_RUNS_PATH)
    clean = alignment.calibration_history(alignment.load_history_from_frame(df))
    effective_session = _next_nymex_business_session(clean["date"].max())
    existing_index = next(
        (i for i, item in enumerate(artifacts)
         if item["effective_session"] == effective_session), None)
    if existing_index is not None:
        existing = artifacts[existing_index]
        # An artifact written by a superseded engine recorded a different
        # training set -- the previous one trained on the whole file, this one
        # only on the alignment-verified era -- so its source hash cannot match
        # and must not be re-verified against the current definition.  It stays
        # in the ledger as the immutable record of what that session actually
        # used.
        if existing.get("candidate_grid_version") != CALIBRATION_METHOD_VERSION:
            print(f"Existing artifact {existing['artifact_id'][:12]} for {effective_session} "
                  f"was produced by '{existing.get('candidate_grid_version')}'; leaving it "
                  f"untouched. New artifacts use '{CALIBRATION_METHOD_VERSION}'.")
            return existing, False

        _, training_df = _eligible_training_history(df, effective_session)
        if (existing["training_end"] != training_df["date"].iloc[-1].date().isoformat()
                or existing["source_row_count"] != len(training_df)
                or existing["source_history_hash"] != _history_hash(training_df)):
            raise ValueError(
                f"Source history no longer matches calibration artifact for {effective_session}.")
        print(f"Verified existing shadow calibration artifact {existing['artifact_id'][:12]} "
              f"for {effective_session} (training through {existing['training_end']}).")
        return existing, False

    prior = artifacts[-1] if artifacts else None
    artifact = build_shadow_calibration_artifact(
        df, cfg, effective_session=effective_session, prior_artifact=prior)
    written, created = append_calibration_artifact(CALIBRATION_RUNS_PATH, artifact)
    action = "Created" if created else "Verified existing"
    print(f"{action} shadow calibration artifact {written['artifact_id'][:12]} "
          f"for {written['effective_session']} (training through {written['training_end']}).")
    return written, created


def main():
    print("Starting pass-through calibration engine...")
    validate_data.validate_all(DATA_DIR)
    cfg = load_config()
    calibration_seed_cfg = dict(cfg)

    if not os.path.exists(CSV_PATH):
        print("No CSV data found. Exiting.")
        sys.exit(0)

    full = alignment.load_history(CSV_PATH)

    # Hard gate.  A one-session stamping error anywhere in the calibration
    # window shows up here as a significant lag-1 pass-through coefficient, and
    # calibrating through it is exactly the failure this engine exists to avoid.
    report = alignment.assert_calibration_alignment(full)
    for prefix in ("RB", "HO"):
        diag = report[prefix]
        print(f"[{prefix}] alignment ok: b0={diag['b0']:+.3f} b1={diag['b1']:+.3f} "
              f"(ratio {diag['ratio']:.2f}, p={diag['p_b1']:.3f}) over {diag['n']} pairs")
    structure = report.get("structure")
    if structure:
        print(f"[--] stamping: {structure['legacy_weeks']}/{structure['total_weeks']} "
              f"weeks legacy-stamped ({structure['fraction']:.0%}); "
              f"limit {alignment.MAX_LEGACY_WEEK_FRACTION:.0%}")

    df = alignment.calibration_history(full)
    cfg["CALIBRATION_ERA_START"] = alignment.CALIBRATION_ERA_START

    min_rows = cfg.get("MIN_ROWS_FOR_TUNING", 120)
    if len(df) < min_rows:
        print(f"Insufficient alignment-verified data. Have {len(df)} rows, need {min_rows}.")
        sys.exit(1)

    source_history_hash = _history_hash(df)
    latest_history_session = df["date"].iloc[-1].date().isoformat()
    effective_session = _next_nymex_business_session(df["date"].iloc[-1])

    if calibration_is_current(cfg, source_history_hash, effective_session,
                              latest_history_session):
        messages = {p: "unchanged history; calibration preserved" for p in ("RB", "HO")}
        print(f"Calibration already applied to history through {latest_history_session}; "
              "skipping recalibration.")
    else:
        messages = {}
        for prefix in ("RB", "HO"):
            cfg, messages[prefix], evaluation = calibrate(df, prefix, cfg)
            print(f"[{prefix}] {messages[prefix]}")
            for fold in evaluation["folds"]:
                print(f"    fold {fold['test_start']}..{fold['test_end']}: "
                      f"{fold['alerts']:3d} alerts, "
                      f"precision {fold['precision']:.1%}, "
                      f"savings {fold['total_savings']:+.1f}c")
        cfg["LAG_DAYS"] = 0

    save_metrics_cache(cfg, effective_session, source_history_hash)
    write_shadow_calibration_artifact(df, calibration_seed_cfg)
    validate_data.validate_calibration_artifacts(CALIBRATION_RUNS_PATH)
    validate_data.validate_and_update_hashes(DATA_DIR)

    local_now = pd.Timestamp.now(tz="America/Chicago")
    commit_msg = (f"Auto-tune [{local_now.strftime('%Y-%m-%d')}]: pass-through. "
                  f"RB({messages['RB']}) HO({messages['HO']})")
    print(commit_msg)
    git_commit_push(commit_msg)


if __name__ == "__main__":
    main()
