"""Read-only chronological comparison of fixed pass-through alternatives.

This reconstructs research forecasts; it does not replay historical production
artifacts or provide a new live track record. Candidates are fixed, with no
search or automatic promotion. The final 90 observations are a diagnostic slice,
not an untouched holdout: this history has already informed model development.
"""

import hashlib
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alignment
import backtest
import model


CANDIDATES = ("current_360", "short_180", "recent_weighted_360")
WINDOW = 360
HALF_LIFE = 90
PURGE = 1
RECENT = 90
BLOCK = 10


def fit_candidate(frame, candidate):
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate}")
    train = frame.tail(180 if candidate == "short_180" else WINDOW)
    x = train["delta_nymex"].to_numpy(dtype=float)
    y = train["delta_rack"].to_numpy(dtype=float)
    if candidate != "recent_weighted_360":
        return model.fit_passthrough(x, y)
    if len(train) < model.MIN_FIT_ROWS or not np.isfinite([x, y]).all():
        raise model.ModelFitError("Insufficient or non-finite weighted training data")
    weights = np.exp2(-np.arange(len(x) - 1, -1, -1) / HALF_LIFE)
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, rank, _ = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)
    if rank != 2:
        raise model.ModelFitError("Weighted design is rank deficient")
    residuals = y - design @ beta
    sse = float(residuals @ residuals)
    total = float(((y - y.mean()) ** 2).sum())
    # Only coefficient fitting is weighted. All three candidates use the
    # production empirical residual calibration over their own fit window.
    return model.PassThrough(
        intercept=float(beta[0]), slope=float(beta[1]), n=len(x),
        sigma=math.sqrt(sse / (len(x) - 2)),
        r2=1.0 - sse / total if total > 0 else 0.0,
        residuals=np.sort(residuals))


def evaluate(frame, target, lean_target, floor):
    """Score each date once; never use test outcomes in their own training."""
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("Expected unique chronological observation dates")
    records = []
    for index in range(WINDOW + PURGE, len(frame)):
        train_end = index - PURGE
        train = frame.iloc[train_end - WINDOW:train_end]
        row = frame.iloc[index]
        if not train["date"].max() < row["date"]:
            raise ValueError("Training includes a test date")
        x, y = float(row["delta_nymex"]), float(row["delta_rack"])
        for candidate in CANDIDATES:
            fit = fit_candidate(train, candidate)
            hike, drop = model.apply_noise_floor(
                *fit.threshold_for_confidence(target), floor)
            lean_hike, lean_drop = model.apply_noise_floor(
                *fit.threshold_for_confidence(lean_target), floor)
            lean_hike, lean_drop = min(lean_hike, hike), max(lean_drop, drop)
            direction = (
                1 if x >= hike else -1 if x <= drop
                else 1 if x >= lean_hike else -1 if x <= lean_drop else 0)
            probability_up = float(fit.probability_up(x))
            probability = float(fit.probability_correct(x))
            # Zero moves cannot count as a correct directional alert. The
            # all-row Brier score consistently forecasts the event rack > 0.
            correct = bool(direction * y > 0) if direction else False
            records.append({
                "date": row["date"], "candidate": candidate,
                "train_end": train["date"].iloc[-1],
                "actual": y, "expected": float(fit.expected_rack_move(x)),
                "direction": direction, "correct": correct,
                "probability": probability, "probability_up": probability_up,
                "payoff": direction * y,
            })
    return pd.DataFrame(records)


def mean_interval(values, bootstrap=2000, seed=1729):
    """Paired moving-block interval retaining short serial dependence."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite paired observations")
    length = min(BLOCK, values.size)
    count = math.ceil(values.size / length)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(bootstrap):
        starts = rng.integers(0, values.size - length + 1, size=count)
        sample = np.concatenate([values[start:start + length] for start in starts])
        means.append(float(sample[:values.size].mean()))
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def summarize(records):
    results = {}
    baseline = records[records["candidate"] == CANDIDATES[0]].set_index("date")
    if len(baseline) < 2:
        raise ValueError("Insufficient evaluation history")
    for candidate in CANDIDATES:
        scored = records[records["candidate"] == candidate].set_index("date")
        if not scored.index.equals(baseline.index):
            raise ValueError("Candidate comparison dates do not match")
        errors = scored["expected"] - scored["actual"]
        active = scored["direction"] != 0
        mse_improvement = ((baseline["expected"] - baseline["actual"]) ** 2
                           - errors ** 2).to_numpy()
        payoff_improvement = (scored["payoff"] - baseline["payoff"]).to_numpy()
        output = {
            "sessions": len(scored),
            "start": scored.index[0].date().isoformat(),
            "end": scored.index[-1].date().isoformat(),
            "mae_cents": float(errors.abs().mean()),
            "rmse_cents": float(np.sqrt((errors ** 2).mean())),
            "brier_up_all_sessions": float(((scored["probability_up"]
                - (scored["actual"] > 0).astype(float)) ** 2).mean()),
            "alerts": int(active.sum()),
            "precision": float(scored.loc[active, "correct"].mean()) if active.any() else None,
            "payoff_cents_per_session": float(scored["payoff"].mean()),
            "payoff_cents_per_alert": float(scored.loc[active, "payoff"].mean()) if active.any() else None,
            "mse_improvement_vs_current": float(mse_improvement.mean()),
            "mse_improvement_95_interval": mean_interval(mse_improvement),
            "payoff_improvement_cents_per_session": float(payoff_improvement.mean()),
            "payoff_improvement_95_interval": mean_interval(payoff_improvement),
        }
        results[candidate] = output
    return results


def run(history_path=ROOT / "data/graves_history.csv"):
    history_path = Path(history_path)
    history = alignment.load_history(str(history_path))
    with redirect_stdout(sys.stderr):
        alignment.assert_calibration_alignment(history)
    history = alignment.calibration_history(history)
    cfg = backtest.load_config()
    output = {
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "method": {
            "candidate_names": list(CANDIDATES), "purge_observations": PURGE,
            "weighted_half_life_observations": HALF_LIFE,
            "bootstrap_block_observations": BLOCK,
            "target": cfg["TARGET_SIGNAL_CONFIDENCE"],
            "lean_target": cfg["TARGET_LEAN_CONFIDENCE"],
            "noise_floor_cents": cfg["SNAPSHOT_NOISE_FLOOR_CENTS"],
            "qualification": (
                "Historical research forecasts; not recorded live results or an untouched "
                "holdout. No candidate search or automatic promotion. Positive paired "
                "improvements favor the candidate. Intervals are exploratory, not "
                "adjusted for multiple comparisons. Payoffs are equal-volume gross "
                "timing proxies, exclude inventory/fees, and are not actual savings. "
                "Each commodity is evaluated separately; RB/HO are never pooled as "
                "independent observations. Lean and full alerts are both scored."),
        },
        "commodities": {},
    }
    for prefix in ("RB", "HO"):
        frame = backtest.training_pairs(history, prefix)
        records = evaluate(frame, cfg["TARGET_SIGNAL_CONFIDENCE"],
                           cfg["TARGET_LEAN_CONFIDENCE"], cfg["SNAPSHOT_NOISE_FLOOR_CENTS"])
        recent_dates = sorted(records["date"].unique())[-RECENT:]
        earlier = records[~records["date"].isin(recent_dates)]
        output["commodities"][prefix] = {
            "all": summarize(records),
            "earlier": summarize(earlier),
            "recent_90": summarize(records[records["date"].isin(recent_dates)]),
        }
    return output


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
