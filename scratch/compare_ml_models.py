"""Offline, prequential ML research; never imported by the live tracker.

All features are known by the decision date under the explicit assumption that
the previous session's rack posting has arrived. Historical publication times
are not available for every row, so this is research rather than a live replay.
Hyperparameters are fixed before comparison; no search or automatic promotion.
"""

import argparse
import hashlib
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alignment
import backtest
import model
from scratch.compare_adaptive_models import mean_interval


WINDOW = 360
BASE_MIN = 120
CORRECTION_MIN = 80
CALIBRATION_WINDOW = 120
PURGE = 1
EVALUATION_START = WINDOW + PURGE
CANDIDATES = (
    "production_ols_360", "prequential_ols_360", "huber_360",
    "ridge_residual", "boosted_residual",
)
CORRECTIONS = CANDIDATES[-2:]
FEATURES = (
    "nymex_move_cents", "abs_nymex_move_cents", "prior_basis_cents",
    "prior_base_error_cents", "prior_base_error_missing",
    "prior_20_move_std_cents", "season_sin", "season_cos",
)
HYPERPARAMETERS = {
    "ridge": {"alpha": 10.0, "solver": "svd"},
    "huber": {"epsilon": 1.35, "alpha": 0.0001,
              "max_iter": 1000, "tol": 1e-6},
    "boosting": {"n_estimators": 60, "max_depth": 2,
                 "min_samples_leaf": 30, "learning_rate": 0.05,
                 "loss": "squared_error", "random_state": 1729},
}


def baseline_forecasts(frame):
    """Generate earlier-only baseline forecasts for residual learning targets."""
    predictions = np.full(len(frame), np.nan)
    fits = {}
    for index in range(BASE_MIN + PURGE, len(frame)):
        end = index - PURGE
        train = frame.iloc[max(0, end - WINDOW):end]
        fit = model.fit_passthrough(train["delta_nymex"], train["delta_rack"])
        predictions[index] = fit.expected_rack_move(frame.iloc[index]["delta_nymex"])
        fits[index] = fit
    return predictions, fits


def causal_features(frame, history, prefix, baseline_predictions):
    """Use exact prior-session joins, never positional shifts across gaps."""
    nymex, rack = alignment.COMMODITY_COLUMNS[prefix]
    prices = history.set_index("date")
    if prices.index.duplicated().any():
        raise ValueError("Feature price history has duplicate dates")
    prior_basis = ((prices[rack] - prices[nymex]) * 100).reindex(
        pd.DatetimeIndex(frame["prior_date"])).to_numpy()
    if not np.isfinite(prior_basis).all():
        raise ValueError("Exact prior-session basis is unavailable")
    errors = pd.Series(
        frame["delta_rack"].to_numpy() - baseline_predictions,
        index=pd.DatetimeIndex(frame["date"]))
    prior_errors = errors.reindex(pd.DatetimeIndex(frame["prior_date"])).to_numpy()
    missing = ~np.isfinite(prior_errors)
    season = 2 * np.pi * (frame["date"].dt.dayofyear.to_numpy() - 1) / 365.25
    features = pd.DataFrame({
        "nymex_move_cents": frame["delta_nymex"].to_numpy(),
        "abs_nymex_move_cents": frame["delta_nymex"].abs().to_numpy(),
        "prior_basis_cents": prior_basis,
        "prior_base_error_cents": np.where(missing, 0.0, prior_errors),
        "prior_base_error_missing": missing.astype(float),
        "prior_20_move_std_cents": frame["delta_nymex"].shift(1).rolling(
            20, min_periods=20).std(ddof=1).to_numpy(),
        "season_sin": np.sin(season), "season_cos": np.cos(season),
    }, index=frame.index)
    return features.loc[:, FEATURES]


def make_correction(candidate):
    if candidate == "ridge_residual":
        return make_pipeline(StandardScaler(), Ridge(**HYPERPARAMETERS["ridge"]))
    if candidate == "boosted_residual":
        return GradientBoostingRegressor(**HYPERPARAMETERS["boosting"])
    raise ValueError(f"Unknown correction candidate: {candidate}")


def fit_checked(estimator, x, y):
    if not np.isfinite(np.asarray(x)).all() or not np.isfinite(np.asarray(y)).all():
        raise ValueError("Non-finite ML training inputs")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x, y)
    return estimator


def residual_distribution(point_prediction, earlier_errors):
    """Empirical probabilities and an 80% interval from earlier OOS errors."""
    errors = np.asarray(earlier_errors, dtype=float)
    if not np.isfinite(point_prediction) or not np.isfinite(errors).all() or not errors.size:
        raise ValueError("Forecast calibration requires finite prior errors")
    p_up = (np.count_nonzero(errors > -point_prediction) + 1) / (errors.size + 2)
    p_down = (np.count_nonzero(errors < -point_prediction) + 1) / (errors.size + 2)
    low, high = np.quantile(errors, [0.1, 0.9]) + point_prediction
    return float(p_up), float(p_down), float(low), float(high)


def probability_action(point_prediction, p_up, p_down, move, floor, target, lean_target):
    if abs(move) < floor or point_prediction == 0:
        return 0, "FLAT"
    probability = p_up if point_prediction > 0 else p_down
    if probability < lean_target:
        return 0, "FLAT"
    return (1 if point_prediction > 0 else -1), ("FULL" if probability >= target else "LEAN")


def production_action(fit, move, floor, target, lean_target):
    hike, drop = model.apply_noise_floor(*fit.threshold_for_confidence(target), floor)
    lean_hike, lean_drop = model.apply_noise_floor(
        *fit.threshold_for_confidence(lean_target), floor)
    if move >= hike:
        return 1, "FULL"
    if move <= drop:
        return -1, "FULL"
    if move >= min(lean_hike, hike):
        return 1, "LEAN"
    if move <= max(lean_drop, drop):
        return -1, "LEAN"
    return 0, "FLAT"


def evaluate(frame, history, prefix, target=0.75, lean_target=0.65, floor=0.6):
    """Daily expanding-origin evaluation with fixed rolling fit/calibration windows."""
    frame = frame.reset_index(drop=True).copy()
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("Expected unique chronological training pairs")
    if not (frame["prior_date"] < frame["date"]).all():
        raise ValueError("Features must reference an earlier session")
    if len(frame) <= EVALUATION_START:
        raise ValueError("Insufficient history for common ML evaluation")
    baseline, fits = baseline_forecasts(frame)
    actual = frame["delta_rack"].to_numpy(dtype=float)
    base_errors = actual - baseline
    features = causal_features(frame, history, prefix, baseline)
    predictions = {name: np.full(len(frame), np.nan) for name in CANDIDATES[1:]}
    predictions["prequential_ols_360"] = baseline.copy()
    records = []
    for index in range(BASE_MIN + PURGE, len(frame)):
        end = index - PURGE
        start = max(0, end - WINDOW)
        x_train = frame.iloc[start:end][["delta_nymex"]].to_numpy()
        y_train = actual[start:end]
        x_test = np.array([[float(frame.iloc[index]["delta_nymex"])]])
        robust = fit_checked(
            make_pipeline(StandardScaler(), HuberRegressor(**HYPERPARAMETERS["huber"])),
            x_train, y_train)
        predictions["huber_360"][index] = robust.predict(x_test)[0]
        correction_indices = np.flatnonzero(np.isfinite(baseline[:end]))[-WINDOW:]
        if correction_indices.size < CORRECTION_MIN:
            continue
        for candidate in CORRECTIONS:
            correction = fit_checked(make_correction(candidate),
                                     features.iloc[correction_indices],
                                     base_errors[correction_indices])
            predictions[candidate][index] = baseline[index] + correction.predict(
                features.iloc[[index]])[0]
        if index < EVALUATION_START:
            continue

        calibration_indices = {}
        for candidate in CANDIDATES[1:]:
            previous = np.flatnonzero(np.isfinite(predictions[candidate][:end]))
            calibration_indices[candidate] = previous[-CALIBRATION_WINDOW:]
        if any(len(indices) < CALIBRATION_WINDOW for indices in calibration_indices.values()):
            raise ValueError("Common evaluation date lacks prior OOS calibration")
        row = frame.iloc[index]
        for candidate in CANDIDATES:
            calibration_name = ("prequential_ols_360" if candidate == CANDIDATES[0]
                                else candidate)
            indices = calibration_indices[calibration_name]
            mean = float(baseline[index] if candidate == CANDIDATES[0]
                         else predictions[candidate][index])
            errors = actual[indices] - predictions[calibration_name][indices]
            p_up, p_down, low, high = residual_distribution(mean, errors)
            if candidate == CANDIDATES[0]:
                # Preserve the current production confidence/threshold logic
                # as a separate control. Its interval uses the common OOS adapter.
                fit = fits[index]
                p_up = float(fit.probability_up(row["delta_nymex"]))
                p_down = float(1.0 - p_up)
                direction, band = production_action(
                    fit, row["delta_nymex"], floor, target, lean_target)
            else:
                direction, band = probability_action(
                    mean, p_up, p_down, row["delta_nymex"], floor, target, lean_target)
            records.append({
                "date": row["date"], "commodity": prefix, "candidate": candidate,
                "train_end": frame.iloc[end - 1]["date"],
                "calibration_end": frame.iloc[indices[-1]]["date"],
                "train_n": end - start,
                "correction_train_n": len(correction_indices) if candidate in CORRECTIONS else 0,
                "calibration_n": len(indices),
                "nymex_move_cents": float(row["delta_nymex"]),
                "actual": float(actual[index]), "point_prediction": mean,
                "p_up": p_up, "p_down": p_down,
                "interval_low": low, "interval_high": high,
                "direction": direction, "band": band,
                "correct": bool(direction * actual[index] > 0) if direction else False,
                "payoff": float(direction * actual[index]),
            })
    return pd.DataFrame(records)


def summarize(records, bootstrap=2000):
    output = {}
    baseline = records[records["candidate"] == CANDIDATES[0]].set_index("date")
    control = records[records["candidate"] == CANDIDATES[1]].set_index("date")
    if len(baseline) < 2:
        raise ValueError("Insufficient scored observations")
    for name in CANDIDATES:
        rows = records[records["candidate"] == name].set_index("date")
        if rows.index.has_duplicates or not rows.index.equals(baseline.index):
            raise ValueError("Models must be scored on identical unique dates")
        error = rows["point_prediction"] - rows["actual"]
        brier = (rows["p_up"] - (rows["actual"] > 0).astype(float)) ** 2
        active = rows["direction"] != 0
        item = {
            "sessions": len(rows), "start": rows.index[0].date().isoformat(),
            "end": rows.index[-1].date().isoformat(),
            "mae_cents": float(error.abs().mean()),
            "rmse_cents": float(np.sqrt((error ** 2).mean())),
            "brier_up_all_sessions": float(brier.mean()),
            "interval_80_coverage": float(((rows["actual"] >= rows["interval_low"])
                & (rows["actual"] <= rows["interval_high"])).mean()),
            "interval_80_mean_width_cents": float((rows["interval_high"] - rows["interval_low"]).mean()),
            "interval_80_mean_score_cents": float((
                rows["interval_high"] - rows["interval_low"]
                + 10 * (rows["interval_low"] - rows["actual"]).clip(lower=0)
                + 10 * (rows["actual"] - rows["interval_high"]).clip(lower=0)).mean()),
            "alerts": int(active.sum()),
            "precision": float(rows.loc[active, "correct"].mean()) if active.any() else None,
            "wait_alerts": int((rows["direction"] == -1).sum()),
            "wrong_waits": int(((rows["direction"] == -1) & (rows["actual"] > 0)).sum()),
            "worst_alert_payoff_cents": float(rows.loc[active, "payoff"].min()) if active.any() else None,
            "payoff_cents_per_session": float(rows["payoff"].mean()),
            "changed_actions_vs_production": int((rows["direction"] != baseline["direction"]).sum()),
        }
        for control_name, benchmark in [("production", baseline), ("same_adapter_ols", control)]:
            losses = (benchmark["point_prediction"] - benchmark["actual"]).abs() - error.abs()
            brier_gain = ((benchmark["p_up"] - (benchmark["actual"] > 0).astype(float)) ** 2 - brier)
            payoff_gain = rows["payoff"] - benchmark["payoff"]
            item[f"vs_{control_name}"] = {
                "mae_improvement_cents": float(losses.mean()),
                "mae_improvement_95_interval": mean_interval(losses, bootstrap=bootstrap),
                "brier_improvement": float(brier_gain.mean()),
                "brier_improvement_95_interval": mean_interval(brier_gain, bootstrap=bootstrap),
                "payoff_improvement_cents_per_session": float(payoff_gain.mean()),
                "payoff_improvement_95_interval": mean_interval(payoff_gain, bootstrap=bootstrap),
            }
        output[name] = item
    return output


def paired_block_intervals(values, block_length, resamples=20000,
                           confidence=1 - 0.05 / 6, seed=1729):
    """Common block draws for paired loss differences across challengers.

    Confidence uses a Bonferroni allocation across three challengers and two
    fuels. The block bootstrap is an approximation under temporal dependence,
    not a guarantee for arbitrary nonstationary series.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Need a finite session-by-challenger loss matrix")
    n = values.shape[0]
    length = min(int(block_length), n)
    if length < 1 or resamples < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid block bootstrap geometry")
    count = int(np.ceil(n / length))
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, resamples, 250):
        batch = min(250, resamples - start)
        starts = rng.integers(0, n - length + 1, size=(batch, count))
        indices = (starts[:, :, None] + np.arange(length)).reshape(batch, -1)[:, :n]
        draws.append(values[indices].mean(axis=1))
    alpha = 1 - confidence
    return np.quantile(np.concatenate(draws), [alpha / 2, 1 - alpha / 2], axis=0).T


def primary_inference(records, resamples=20000):
    """Full-period MAE, with six-comparison correction and block sensitivity."""
    baseline = records[records["candidate"] == CANDIDATES[0]].set_index("date")
    base_loss = (baseline["point_prediction"] - baseline["actual"]).abs()
    challengers = ("huber_360", "ridge_residual", "boosted_residual")
    gains = []
    for name in challengers:
        rows = records[records["candidate"] == name].set_index("date")
        if not rows.index.equals(baseline.index):
            raise ValueError("Primary inference requires identical paired dates")
        gains.append((base_loss - (rows["point_prediction"] - rows["actual"]).abs()).to_numpy())
    matrix = np.column_stack(gains)
    intervals = {length: paired_block_intervals(matrix, length, resamples=resamples)
                 for length in (5, 10, 20)}
    return {
        name: {
            "mae_improvement_cents": float(matrix[:, column].mean()),
            "bonferroni_6_comparison_intervals": {
                str(length): [float(v) for v in bounds[column]]
                for length, bounds in intervals.items()},
            "positive_lower_bound_at_all_block_lengths": bool(
                all(bounds[column, 0] > 0 for bounds in intervals.values())),
        }
        for column, name in enumerate(challengers)
    }


def run(history_path=ROOT / "data/graves_history.csv"):
    history_path = Path(history_path)
    history = alignment.load_history(str(history_path))
    with redirect_stdout(sys.stderr):
        alignment.assert_calibration_alignment(history)
    history = alignment.calibration_history(history)
    cfg = backtest.load_config()
    output = {
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "research_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("model.py", "alignment.py", "backtest.py", "futures_util.py",
                         "scratch/compare_adaptive_models.py")},
        "method": {
            "candidates": list(CANDIDATES), "features": list(FEATURES),
            "hyperparameters": HYPERPARAMETERS,
            "rolling_fit_observations": WINDOW, "base_warmup_observations": BASE_MIN,
            "correction_min_observations": CORRECTION_MIN, "purge_observations": PURGE,
            "prequential_calibration_observations": CALIBRATION_WINDOW,
            "bootstrap_block_observations": 10, "bootstrap_resamples": 2000,
            "target_confidence": cfg["TARGET_SIGNAL_CONFIDENCE"],
            "lean_confidence": cfg["TARGET_LEAN_CONFIDENCE"],
            "noise_floor_cents": cfg["SNAPSHOT_NOISE_FLOOR_CENTS"],
            "primary_inference": {
                "loss": "Full-period paired absolute point-forecast error",
                "family": "Three ML challengers versus production OLS, separately for RB and HO",
                "comparisons": 6, "family_alpha": 0.05,
                "per_comparison_confidence": 1 - 0.05 / 6,
                "block_sensitivity_observations": [5, 10, 20],
                "bootstrap_resamples": 20000,
            },
            "qualification": (
                "Reconstructed historical forecasts, not live performance or an untouched "
                "holdout. Assumes prior-session rack postings were available by the next "
                "decision. Roll exclusions inherit the production calendar-based filter. "
                "Correction targets and calibration errors come only from earlier "
                "out-of-sample forecasts. All-row probability target is rack_change > 0; "
                "zero changes are incorrect directional alerts. Production control retains "
                "its original in-sample-residual probability/threshold method; the OLS "
                "adapter control uses the same OOS calibration and action rule as ML. "
                "All intervals use earlier OOS errors and are empirical, not guaranteed "
                "conditional coverage. Full-period primary MAE intervals use a six-test "
                "Bonferroni correction and block-length sensitivity; these bootstrap "
                "intervals are approximate under dependence. Secondary and subperiod "
                "intervals are exploratory and unadjusted for multiple comparisons. "
                "Huber estimates a robust location, not a guaranteed conditional mean. "
                "Gross timing proxies assume equal volume, "
                "exclude inventory/fees, and are not actual savings. No automatic promotion."),
        },
        "commodities": {},
    }
    all_records = []
    for prefix in ("RB", "HO"):
        print(f"Evaluating {prefix} ML candidates...", file=sys.stderr, flush=True)
        frame = backtest.training_pairs(history, prefix)
        records = evaluate(frame, history, prefix,
                           target=cfg["TARGET_SIGNAL_CONFIDENCE"],
                           lean_target=cfg["TARGET_LEAN_CONFIDENCE"],
                           floor=cfg["SNAPSHOT_NOISE_FLOOR_CENTS"])
        recent_dates = sorted(records["date"].unique())[-90:]
        output["commodities"][prefix] = {
            "all": summarize(records),
            "earlier": summarize(records[~records["date"].isin(recent_dates)]),
            "recent_90": summarize(records[records["date"].isin(recent_dates)]),
            "primary_inference": primary_inference(records),
        }
        all_records.append(records)
    return output, pd.concat(all_records, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/graves_history.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional directory under reports/ for summary and research forecasts")
    args = parser.parse_args()
    if args.output_dir and not args.output_dir.resolve().is_relative_to(ROOT / "reports"):
        parser.error("Output directory must be inside the repository reports/ directory")
    result, records = run(args.history)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        last_date = records["date"].max().date().isoformat()
        stem = args.output_dir / f"ml_model_comparison_{last_date}"
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        records.to_csv(str(stem) + "_forecasts.csv", index=False, float_format="%.10g")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
