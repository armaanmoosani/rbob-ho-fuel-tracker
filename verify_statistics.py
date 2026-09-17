"""Model validation suite.

Every check here must be capable of failing.  The previous version had four
that were not:

* the "null model test" passed when the mean R-squared of shuffled data was
  below 0.01, which it always is (it is approximately 1/n);
* "residual diagnostics" passed when the mean OLS residual was below 1e-5,
  which it always is when the fit has an intercept;
* "sensitivity analysis" fitted thresholds on the full history and scored them
  on the full history, then reported the result as robustness;
* "shadow benchmarks" compared against a supposed 50/50 coin flip that was
  actually a uniform three-way choice with an expectation of zero, so beating
  it meant only "savings above zero".

It also seeded nothing, so results changed between runs, quoted permutation
p-values of exactly 0.0000, and overwrote a committed PNG as a side effect.

Usage:
    python3 verify_statistics.py            # report only
    python3 verify_statistics.py --chart    # also write the PNG
"""

import argparse
import os
import sys

import numpy as np

import alignment
import model
from futures_util import is_contract_roll_day

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

SEED = 20260917
N_PERMUTATIONS = 5000
HOLDOUT_ROWS = 60


class Check:
    """One pass/fail assertion with the evidence behind it."""

    def __init__(self, name, passed, detail):
        self.name = name
        self.passed = bool(passed)
        self.detail = detail

    def __str__(self):
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}\n         {self.detail}"


def permutation_p_value(observed, null_samples):
    """``(k + 1) / (n + 1)``.

    The +1 is not cosmetic: ``mean(null >= observed)`` can return exactly 0,
    and a permutation test cannot licence a p-value of zero -- the most it can
    say with n draws is ``1/(n+1)``.
    """
    null_samples = np.asarray(null_samples, dtype=float)
    at_least_as_extreme = int(np.sum(null_samples >= observed))
    return (at_least_as_extreme + 1) / (len(null_samples) + 1)


def pairs_for(prefix, df=None):
    frame = alignment.aligned_deltas(
        df if df is not None else alignment.calibration_history(), prefix)
    keep = ~frame["date"].apply(lambda d: is_contract_roll_day(d.date(), prefix))
    return frame[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_alignment(checks):
    """The calibration window must be same-session aligned."""
    full = alignment.load_history()
    try:
        report = alignment.assert_calibration_alignment(full)
    except alignment.AlignmentError as exc:
        checks.append(Check("History alignment", False, str(exc)))
        return
    detail = "; ".join(
        f"{p}: b0={report[p]['b0']:+.3f} b1={report[p]['b1']:+.3f} "
        f"(ratio {report[p]['ratio']:.2f}, p={report[p]['p_b1']:.3f})"
        for p in ("RB", "HO"))
    checks.append(Check("History alignment", True, detail))

    # The detector must still catch the defect it was built for.  The live file
    # has been re-dated, so the archived pre-migration copy is the positive
    # case; if that ever stops being flagged, the detector is broken.
    archive = os.path.join(DATA_DIR, "graves_history.pre_alignment_migration.csv")
    if os.path.exists(archive):
        before = alignment.load_history(archive)
        rejected = [p for p in ("RB", "HO")
                    if not alignment.lag_diagnostics(before, p)["aligned"]]
        checks.append(Check(
            "Detector still flags the known-bad pre-migration file",
            bool(rejected),
            f"archived original rejected for {rejected or 'nothing -- DETECTOR IS BROKEN'}"))
    else:
        checks.append(Check(
            "Detector still flags the known-bad pre-migration file", False,
            "pre-migration archive is missing; the detector has no positive case"))


def check_holdout(prefix, checks):
    """Out-of-sample holdout against a permutation null.

    The holdout is genuinely frozen: the model is fitted only on rows before
    it, and no hyper-parameter is read from a cache that saw the whole series.
    """
    frame = pairs_for(prefix)
    if len(frame) < HOLDOUT_ROWS + model.MIN_FIT_ROWS:
        checks.append(Check(f"{prefix} frozen holdout", False, "insufficient history"))
        return None

    train = frame.iloc[:-HOLDOUT_ROWS]
    test = frame.iloc[-HOLDOUT_ROWS:]
    fit = model.fit_passthrough(train["delta_nymex"], train["delta_rack"])
    hike, drop = model.apply_noise_floor(*fit.threshold_for_confidence(0.75), 1.2)
    observed = model.summarize(test["delta_nymex"], test["delta_rack"], hike, drop)

    rng = np.random.default_rng(SEED)
    nymex = test["delta_nymex"].to_numpy()
    rack = test["delta_rack"].to_numpy()
    null = np.array([
        model.summarize(nymex, rng.permutation(rack), hike, drop)["total_savings"]
        for _ in range(N_PERMUTATIONS)])
    p_value = permutation_p_value(observed["total_savings"], null)

    checks.append(Check(
        f"{prefix} frozen holdout beats a permutation null",
        p_value < 0.05,
        f"{observed['alerts']} alerts, precision {observed['precision']:.1%}, "
        f"savings {observed['total_savings']:+.1f}c, p={p_value:.4f} "
        f"(null mean {null.mean():+.1f}c)"))
    return {"observed": observed, "null": null, "p_value": p_value}


def check_calibration(prefix, checks):
    """Quoted probabilities must match realised frequencies out of sample."""
    frame = pairs_for(prefix)
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

    if len(quoted) < 60:
        checks.append(Check(f"{prefix} probability calibration", False, "too few points"))
        return None

    quoted = np.array(quoted)
    realised = np.array(realised, dtype=float)
    gap = realised.mean() - quoted.mean()
    base = realised.mean()
    brier = float(np.mean((quoted - realised) ** 2))
    reference = float(np.mean((base - realised) ** 2))
    skill = 1 - brier / reference if reference > 0 else 0.0

    checks.append(Check(
        f"{prefix} probability calibration",
        abs(gap) < 0.06 and brier <= reference * 1.02,
        f"n={len(quoted)} quoted {quoted.mean():.3f} vs realised {realised.mean():.3f} "
        f"(gap {gap:+.3f}), Brier {brier:.4f} vs constant-rate {reference:.4f} "
        f"(skill {skill:+.3f})"))
    return {"quoted": quoted, "realised": realised}


def check_passthrough_is_economically_sane(prefix, checks):
    """The fitted relationship must look like rack pricing, not curve fitting."""
    frame = pairs_for(prefix)
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    sane = 0.2 < fit.slope < 1.3 and fit.r2 > 0.3
    checks.append(Check(
        f"{prefix} pass-through is economically plausible",
        sane,
        f"slope {fit.slope:.3f} (expect 0.2-1.3 for a formula-priced rack), "
        f"intercept {fit.intercept:+.3f}c, R2 {fit.r2:.3f}, n={fit.n}"))
    return fit


def check_residual_autocorrelation(prefix, checks):
    """Residuals must not be strongly serially correlated.

    Unlike the old mean-residual check, this one can fail: the mean of an OLS
    residual is zero by construction, but its autocorrelation is not.
    """
    frame = pairs_for(prefix)
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    resid = frame["delta_rack"].to_numpy() - fit.expected_rack_move(frame["delta_nymex"].to_numpy())
    lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    durbin_watson = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
    checks.append(Check(
        f"{prefix} residuals are not serially correlated",
        abs(lag1) < 0.25,
        f"lag-1 autocorrelation {lag1:+.3f}, Durbin-Watson {durbin_watson:.3f} "
        f"(2.0 = independent)"))


def check_out_of_sample_beats_in_sample_honestly(prefix, checks):
    """In-sample performance must not be reported as out-of-sample.

    Fits on the full series and on a rolling origin are both computed; the
    check is that the gap is disclosed, not that it is zero.
    """
    frame = pairs_for(prefix)
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    hike, drop = model.apply_noise_floor(*fit.threshold_for_confidence(0.75), 1.2)
    in_sample = model.summarize(frame["delta_nymex"], frame["delta_rack"], hike, drop)

    alerts = correct = 0
    total = 0.0
    for start in range(120, len(frame), 10):
        train = frame.iloc[max(0, start - 180):start]
        test = frame.iloc[start:start + 10]
        if len(train) < model.MIN_FIT_ROWS or test.empty:
            continue
        f = model.fit_passthrough(train["delta_nymex"], train["delta_rack"])
        h, d = model.apply_noise_floor(*f.threshold_for_confidence(0.75), 1.2)
        s = model.summarize(test["delta_nymex"], test["delta_rack"], h, d)
        alerts += s["alerts"]
        correct += s["correct"]
        total += s["total_savings"]

    oos_precision = correct / alerts if alerts else float("nan")
    optimism = in_sample["precision"] - oos_precision
    checks.append(Check(
        f"{prefix} in-sample optimism is small",
        abs(optimism) < 0.10,
        f"in-sample {in_sample['precision']:.1%} on {in_sample['alerts']} alerts vs "
        f"out-of-sample {oos_precision:.1%} on {alerts} alerts "
        f"(optimism {optimism:+.1%})"))


def check_benchmarks(prefix, checks):
    """Compare against baselines an operator could actually run.

    The comparison is per *decision*, not per period.  Total savings favour
    whichever strategy acts most often, but a buyer can only dispatch so many
    trucks, so the operative question is what each decision is worth.  The old
    suite compared total savings against a strategy whose expectation was zero,
    which reduced to "savings above zero".

    Note what this reveals: on the days the model fires, it agrees with simple
    sign-following almost exactly.  The threshold's contribution is selectivity
    -- declining the small moves -- not a different call on the big ones.
    """
    frame = pairs_for(prefix)
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    hike, drop = model.apply_noise_floor(*fit.threshold_for_confidence(0.75), 1.2)
    nymex = frame["delta_nymex"].to_numpy()
    rack = frame["delta_rack"].to_numpy()

    active = (nymex >= hike) | (nymex <= drop)
    n_active = int(active.sum())
    if n_active == 0:
        checks.append(Check(f"{prefix} beats naive baselines", False, "no alerts fired"))
        return

    def economics(mask, side):
        payoff = np.where(side, rack, -rack)[mask]
        return {
            "n": int(mask.sum()),
            "per_decision": float(payoff.mean()) if mask.sum() else float("nan"),
            "precision": float((payoff > 0).mean()) if mask.sum() else float("nan"),
            "total": float(payoff.sum()),
        }

    model_stats = economics(active, nymex >= hike)
    # Sign-following acts on EVERY session, which is the honest version of the
    # baseline: it needs no model, so it also needs no threshold.
    all_days = np.ones_like(active, dtype=bool)
    sign_stats = economics(all_days, nymex >= 0)

    rng = np.random.default_rng(SEED)
    random_draws = np.array([
        np.where(rng.random(n_active) < 0.5, rack[active], -rack[active]).sum()
        for _ in range(2000)])
    p_vs_random = permutation_p_value(model_stats["total"], random_draws)

    better_per_decision = model_stats["per_decision"] > sign_stats["per_decision"]
    checks.append(Check(
        f"{prefix} earns more per decision than acting every day",
        better_per_decision and p_vs_random < 0.05,
        f"model {model_stats['per_decision']:+.2f}c/decision at "
        f"{model_stats['precision']:.1%} over {model_stats['n']} decisions | "
        f"sign-following every session {sign_stats['per_decision']:+.2f}c/decision at "
        f"{sign_stats['precision']:.1%} over {sign_stats['n']} | "
        f"coin flip on the same days p={p_vs_random:.4f}"))

    checks.append(Check(
        f"{prefix} selectivity is where the threshold earns its keep "
        f"(informational)",
        True,
        f"the model declines {sign_stats['n'] - model_stats['n']} of "
        f"{sign_stats['n']} sessions; on the ones it takes it agrees with "
        f"sign-following, so the edge is the pass-through itself, not the "
        f"direction call"))


def check_regime_sensitivity(prefix, checks):
    """State plainly how much of the measured edge depends on the regime."""
    frame = pairs_for(prefix)
    volatility = frame["delta_nymex"].abs()
    median_volatility = float(volatility.median())
    fit = model.fit_passthrough(frame["delta_nymex"], frame["delta_rack"])
    hike, drop = model.apply_noise_floor(*fit.threshold_for_confidence(0.75), 1.2)

    rows = []
    for label, mask in (("calm", volatility <= median_volatility),
                        ("volatile", volatility > median_volatility)):
        subset = frame[mask]
        s = model.summarize(subset["delta_nymex"], subset["delta_rack"], hike, drop)
        rows.append((label, s))

    calm, volatile = rows[0][1], rows[1][1]
    detail = (f"calm half: {calm['alerts']} alerts at {calm['precision']:.1%}, "
              f"{calm['mean_savings']:+.2f}c/alert | "
              f"volatile half: {volatile['alerts']} alerts at {volatile['precision']:.1%}, "
              f"{volatile['mean_savings']:+.2f}c/alert")
    # This is informational: it always "passes", but it prints the number an
    # operator needs in order to discount the headline figure.
    checks.append(Check(f"{prefix} regime split (informational)", True, detail))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model validation suite")
    parser.add_argument("--chart", action="store_true",
                        help="write reports/statistical_verification.png "
                             "(off by default so a report run never mutates a "
                             "committed artifact)")
    args = parser.parse_args()

    print("=== MODEL VALIDATION SUITE ===")
    print(f"seed={SEED}  permutations={N_PERMUTATIONS}  "
          f"era>={alignment.CALIBRATION_ERA_START}\n")

    checks = []
    check_alignment(checks)

    holdouts, calibrations = {}, {}
    for prefix in ("RB", "HO"):
        print(f"--- {prefix} ---")
        check_passthrough_is_economically_sane(prefix, checks)
        holdouts[prefix] = check_holdout(prefix, checks)
        calibrations[prefix] = check_calibration(prefix, checks)
        check_residual_autocorrelation(prefix, checks)
        check_out_of_sample_beats_in_sample_honestly(prefix, checks)
        check_benchmarks(prefix, checks)
        check_regime_sensitivity(prefix, checks)
        print()

    print("=== RESULTS ===")
    for check in checks:
        print(check)

    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")

    if args.chart:
        _write_chart(holdouts, calibrations)

    if failed:
        print("FAILING CHECKS:")
        for check in failed:
            print(f"  - {check.name}")
        sys.exit(1)
    print("All checks passed.")


def _write_chart(holdouts, calibrations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#ffffff")

    for ax, prefix in zip(axes[0], ("RB", "HO")):
        result = holdouts.get(prefix)
        if result:
            ax.hist(result["null"], bins=40, color="#94a3b8", edgecolor="none")
            ax.axvline(result["observed"]["total_savings"], color="#ef4444",
                       linestyle="--", linewidth=2,
                       label=f"observed {result['observed']['total_savings']:+.0f}c "
                             f"(p={result['p_value']:.4f})")
            ax.legend(fontsize=8)
        ax.set_title(f"{prefix}: frozen holdout vs permutation null", fontsize=11)
        ax.set_xlabel("cumulative savings (c/gal)", fontsize=9)

    for ax, prefix in zip(axes[1], ("RB", "HO")):
        data = calibrations.get(prefix)
        if data:
            quoted, realised = data["quoted"], data["realised"]
            edges = np.linspace(0.5, 1.0, 6)
            xs, ys, ns = [], [], []
            for low, high in zip(edges[:-1], edges[1:]):
                mask = (quoted >= low) & (quoted < high)
                if mask.sum() >= 5:
                    xs.append(quoted[mask].mean())
                    ys.append(realised[mask].mean())
                    ns.append(int(mask.sum()))
            ax.plot([0.5, 1.0], [0.5, 1.0], color="#94a3b8", linestyle="--",
                    label="perfect calibration")
            ax.scatter(xs, ys, s=[max(25, n * 3) for n in ns], color="#3b82f6",
                       zorder=3, label="observed (area = n)")
            ax.legend(fontsize=8)
        ax.set_title(f"{prefix}: quoted probability vs realised", fontsize=11)
        ax.set_xlabel("quoted", fontsize=9)
        ax.set_ylabel("realised", fontsize=9)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "statistical_verification.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nChart written to {path}")


if __name__ == "__main__":
    main()
