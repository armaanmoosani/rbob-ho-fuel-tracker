"""The rack pass-through model: one implementation shared by every consumer.

The rack is not forecast in any deep sense.  Graves prices off the 1:30 PM CT
NYMEX settle, so the daily rack change is close to an affine function of the
daily settle change:

    rack_delta = a + b * nymex_delta + e,   e ~ N(0, sigma^2)

Everything the system says about a signal derives from that one fit:

* the trigger threshold, placed where the win probability reaches a target
  rather than at an arbitrary percentile of past hike days;
* the confidence quoted to the user, which is a calibrated probability for
  *today's* move rather than a lookup in a coarse Z-score bin;
* the expected size of the move and the tail risk of deferring.

Why this replaced the percentile thresholds
-------------------------------------------
The previous rule took the ``Hp``-th percentile of NYMEX moves on days that
happened to be hikes.  That number has no probabilistic meaning: it cannot say
how likely the next signal is to be right, so the system had to quote a
hardcoded "53%-73%" range copied from a stale README table, and grade
confidence with Z-score bins whose differences were not statistically
significant for RB (high vs low p = 0.37).

Both rules perform the same out of sample on this data -- the objective surface
is almost flat -- so the change is made for interpretability and honesty, not
for a performance gain.  See ``test_model.py`` for the calibration check that
holds the quoted probabilities to their realised frequencies.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Absolute clamps on any threshold the model produces.  These are guardrails
# against a degenerate fit (near-zero slope, collapsed residual variance), not
# tuning knobs.
MIN_ABS_THRESHOLD = 0.3
MAX_ABS_THRESHOLD = 8.0

# Minimum pairs required before a fit is trusted.
MIN_FIT_ROWS = 40


class ModelFitError(RuntimeError):
    """Raised when the pass-through relationship cannot be fitted."""


@dataclass(frozen=True)
class PassThrough:
    """A fitted rack pass-through relationship for one commodity.

    Probabilities and thresholds come from the *empirical* distribution of the
    fit residuals, not from a normal approximation.  The residuals are strongly
    leptokurtic -- excess kurtosis 13 (RB) and 18 (HO), Jarque-Bera p ~ 0 -- so
    a few large rack surprises inflate the standard deviation and drag every
    normal-based probability toward 0.5.  Measured out of sample on an
    expanding origin, the normal version quoted a mean 0.856 against a realised
    0.932 for RB, a 7.6-point under-confidence with *negative* Brier skill.
    Using the empirical residual CDF closes the gap to 3.3 points for RB and
    0.1 points for HO, and turns the skill positive for both.
    """

    intercept: float
    slope: float
    sigma: float
    n: int
    r2: float
    residuals: np.ndarray = None

    # --- residual distribution -------------------------------------------

    def _residual_cdf(self, value):
        """Empirical P(residual <= value), interpolated and never 0 or 1.

        Clipped to ``1/(n+2)`` so the model can never quote certainty from a
        finite sample -- with 240 rows the most it will ever claim is 99.6%.
        """
        if self.residuals is None or self.residuals.size == 0:
            return float(stats.norm.cdf(value / self.sigma)) if self.sigma > 0 else 0.5
        sorted_residuals = self.residuals
        rank = float(np.searchsorted(sorted_residuals, value, side="right"))
        p = rank / sorted_residuals.size
        floor = 1.0 / (sorted_residuals.size + 2)
        return float(np.clip(p, floor, 1.0 - floor))

    def _residual_quantile(self, q):
        """Empirical residual quantile, falling back to the normal fit."""
        if self.residuals is None or self.residuals.size == 0:
            return float(stats.norm.ppf(q) * self.sigma)
        return float(np.quantile(self.residuals, q))

    # --- public interface -------------------------------------------------

    def expected_rack_move(self, nymex_delta):
        """E[rack_delta | nymex_delta], in cents."""
        return self.intercept + self.slope * np.asarray(nymex_delta, dtype=float)

    def probability_up(self, nymex_delta):
        """P(rack_delta > 0 | nymex_delta) = P(residual > -(a + b*x))."""
        mu = np.atleast_1d(self.expected_rack_move(nymex_delta))
        out = np.array([1.0 - self._residual_cdf(-m) for m in mu])
        return out if np.ndim(nymex_delta) else float(out[0])

    def probability_correct(self, nymex_delta):
        """P(the rack moves the way a signal of this size implies).

        For a positive move that is P(rack rises); for a negative move
        P(rack falls).
        """
        x = np.asarray(nymex_delta, dtype=float)
        p_up = np.atleast_1d(self.probability_up(x))
        result = np.where(np.atleast_1d(x) >= 0, p_up, 1.0 - p_up)
        return result if np.ndim(nymex_delta) else float(result[0])

    def threshold_for_confidence(self, target_confidence):
        """NYMEX move at which ``probability_correct`` reaches the target.

        Hike side: we need ``P(residual > -(a + b*t)) = p``, i.e.
        ``-(a + b*t) = Q(1 - p)``.  Drop side: ``P(residual < -(a + b*t)) = p``,
        i.e. ``-(a + b*t) = Q(p)``.

        The two are deliberately not mirror images.  A non-zero intercept means
        the rack drifts even on a flat NYMEX, so one side needs a larger move
        than the other to reach the same confidence -- and the skewed residual
        distribution (RB skew +1.85) widens that difference further.
        """
        if not 0.5 < target_confidence < 1.0:
            raise ValueError("target_confidence must be strictly between 0.5 and 1.0")
        if self.slope <= 0:
            raise ModelFitError(
                f"non-positive pass-through slope ({self.slope:.4f}); refusing to "
                "derive thresholds from an inverted relationship"
            )
        hike = (-self._residual_quantile(1.0 - target_confidence) - self.intercept) / self.slope
        drop = (-self._residual_quantile(target_confidence) - self.intercept) / self.slope
        hike = float(np.clip(hike, MIN_ABS_THRESHOLD, MAX_ABS_THRESHOLD))
        drop = float(np.clip(drop, -MAX_ABS_THRESHOLD, -MIN_ABS_THRESHOLD))
        return hike, drop


def fit_passthrough(delta_nymex, delta_rack):
    """Least-squares fit of the pass-through relationship.

    ``sigma`` uses the residual degrees of freedom (n - 2) so it is unbiased;
    using ``np.std`` directly would understate the spread and therefore
    understate every quoted probability's uncertainty.
    """
    x = np.asarray(delta_nymex, dtype=float)
    y = np.asarray(delta_rack, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < MIN_FIT_ROWS:
        raise ModelFitError(f"only {n} usable pairs; need {MIN_FIT_ROWS}")
    if np.ptp(x) == 0:
        raise ModelFitError("NYMEX moves are constant; cannot fit a slope")

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    dof = n - 2
    sigma = float(np.sqrt(float(resid @ resid) / dof))
    total = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / total) if total > 0 else 0.0
    return PassThrough(intercept=float(intercept), slope=float(slope),
                       sigma=sigma, n=n, r2=r2,
                       residuals=np.sort(resid))


# Number of residual quantiles persisted to the metrics cache.  101 points on a
# 1% grid reproduces the empirical CDF to well under a percentage point of
# probability while keeping the cache small and diffable.
RESIDUAL_QUANTILE_POINTS = 101


def passthrough_to_config(fit, prefix):
    """Serialise a fit into flat, JSON-safe metrics-cache keys.

    The residual distribution travels with the fit as a quantile grid.  Without
    it the live path would have to fall back on the normal approximation and
    would quote systematically under-confident probabilities -- the exact defect
    the empirical CDF was introduced to remove.
    """
    grid = np.linspace(0.0, 1.0, RESIDUAL_QUANTILE_POINTS)
    quantiles = (np.quantile(fit.residuals, grid) if fit.residuals is not None
                 and fit.residuals.size else np.zeros_like(grid))
    return {
        f"{prefix}_pt_intercept": round(float(fit.intercept), 6),
        f"{prefix}_pt_slope": round(float(fit.slope), 6),
        f"{prefix}_pt_sigma": round(float(fit.sigma), 6),
        f"{prefix}_pt_r2": round(float(fit.r2), 6),
        f"{prefix}_pt_n": int(fit.n),
        f"{prefix}_pt_residual_quantiles": [round(float(v), 4) for v in quantiles],
    }


def passthrough_from_config(cfg, prefix):
    """Rebuild a fit from metrics-cache keys, or None when absent."""
    required = (f"{prefix}_pt_intercept", f"{prefix}_pt_slope",
                f"{prefix}_pt_sigma", f"{prefix}_pt_n")
    if any(key not in cfg for key in required):
        return None
    quantiles = cfg.get(f"{prefix}_pt_residual_quantiles") or []
    residuals = np.sort(np.asarray(quantiles, dtype=float)) if quantiles else None
    return PassThrough(
        intercept=float(cfg[f"{prefix}_pt_intercept"]),
        slope=float(cfg[f"{prefix}_pt_slope"]),
        sigma=float(cfg[f"{prefix}_pt_sigma"]),
        n=int(cfg[f"{prefix}_pt_n"]),
        r2=float(cfg.get(f"{prefix}_pt_r2", float("nan"))),
        residuals=residuals,
    )


def apply_noise_floor(hike, drop, noise_floor):
    """Raise thresholds so a signal cannot be triggered by measurement noise.

    The live verdict is computed from a 1:30 PM snapshot, not from the official
    settle the model was fitted on.  On non-roll sessions those differ with a
    robust sigma of about 0.5 cents and a 95th percentile near 1.2 cents.  A
    threshold below that is fired by the measurement error itself, so the floor
    is a hard correctness requirement, not a conservatism preference.
    """
    floor = abs(float(noise_floor))
    return max(float(hike), floor), min(float(drop), -floor)


def economic_decision_value(action, expected_rack_move_cents, gallons,
                            buy_cost_cents=0.0, wait_cost_cents=0.0):
    """Expected procurement value after action-specific operating costs.

    ``expected_rack_move_cents`` is tomorrow's rack minus today's rack.  Buying
    today captures a rise; waiting captures a fall.  Dispatch/carrying costs
    and deferral/stockout costs are deliberately separate because treating
    them as symmetric would encode the wrong physical decision.

    Costs are explicit policy inputs, never estimated from price history.  A
    zero default therefore means "unknown/not configured", not "free".  It
    preserves the existing probability policy while making the economic gate
    operational as soon as the owner supplies real costs.
    """
    try:
        expected = float(expected_rack_move_cents)
        volume = float(gallons)
        buy_cost = float(buy_cost_cents)
        wait_cost = float(wait_cost_cents)
    except (TypeError, ValueError) as exc:
        raise ValueError("economic decision inputs must be numeric") from exc
    if not all(np.isfinite(v) for v in (expected, volume, buy_cost, wait_cost)):
        raise ValueError("economic decision inputs must be finite")
    if volume <= 0:
        raise ValueError("gallons must be positive")
    if buy_cost < 0 or wait_cost < 0:
        raise ValueError("action costs cannot be negative")

    if action in ("BUY_NOW", "LEAN_BUY"):
        gross = expected
        cost = buy_cost
    elif action in ("WAIT", "LEAN_WAIT"):
        gross = -expected
        cost = wait_cost
    else:
        gross = 0.0
        cost = 0.0
    net = gross - cost
    return {
        "gross_edge_cents": gross,
        "action_cost_cents": cost,
        "net_edge_cents": net,
        "net_value_dollars": net / 100.0 * volume,
        "economically_positive": net > 0.0,
    }


def threshold_uncertainty(delta_nymex, delta_rack, target_confidence,
                          noise_floor, bootstrap=500, block_length=5,
                          seed=20260923):
    """Moving-block bootstrap interval for both decision thresholds.

    Daily residuals are not assumed independent.  Resampling short contiguous
    blocks preserves local volatility clustering while refitting both the
    pass-through slope and empirical residual distribution on every draw.
    The returned interval describes calibration uncertainty, not a confidence
    interval for today's realised rack move.
    """
    x = np.asarray(delta_nymex, dtype=float)
    y = np.asarray(delta_rack, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < MIN_FIT_ROWS:
        raise ModelFitError(f"only {n} usable pairs; need {MIN_FIT_ROWS}")
    if bootstrap < 100:
        raise ValueError("bootstrap must be at least 100")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must be between 1 and the sample size")

    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n / block_length))
    max_start = n - block_length
    hikes, drops = [], []
    for _ in range(int(bootstrap)):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        indices = np.concatenate([
            np.arange(start, start + block_length) for start in starts
        ])[:n]
        try:
            fit = fit_passthrough(x[indices], y[indices])
            hike, drop = apply_noise_floor(
                *fit.threshold_for_confidence(target_confidence), noise_floor)
        except ModelFitError:
            continue
        hikes.append(hike)
        drops.append(drop)

    minimum_success = max(80, int(bootstrap * 0.8))
    if len(hikes) < minimum_success:
        raise ModelFitError(
            f"only {len(hikes)} of {bootstrap} bootstrap refits succeeded")
    hike_low, hike_high = np.percentile(hikes, [2.5, 97.5])
    drop_low, drop_high = np.percentile(drops, [2.5, 97.5])
    return {
        "hike_low": float(hike_low),
        "hike_high": float(hike_high),
        "drop_low": float(drop_low),
        "drop_high": float(drop_high),
        "bootstrap": len(hikes),
        "block_length": int(block_length),
    }


def savings_from_signals(delta_nymex, delta_rack, hike, drop):
    """Per-alert procurement payoff, in cents per gallon.

    BUY on a hike signal: you lift at yesterday's rack, so you save the rise.
    WAIT on a drop signal: you defer, so you save the fall.
    Anything between the thresholds is not an alert and contributes nothing.
    """
    x = np.asarray(delta_nymex, dtype=float)
    y = np.asarray(delta_rack, dtype=float)
    buy = x >= hike
    wait = x <= drop
    payoff = np.concatenate([y[buy], -y[wait]])
    correct = int((y[buy] > 0).sum() + (y[wait] < 0).sum())
    return payoff, correct


def summarize(delta_nymex, delta_rack, hike, drop):
    """Alert count, precision and savings for one threshold pair."""
    payoff, correct = savings_from_signals(delta_nymex, delta_rack, hike, drop)
    alerts = int(payoff.size)
    return {
        "alerts": alerts,
        "correct": correct,
        "precision": (correct / alerts) if alerts else float("nan"),
        "total_savings": float(payoff.sum()),
        "mean_savings": float(payoff.mean()) if alerts else float("nan"),
        "payoff": payoff,
    }


def wait_tail_risk(delta_nymex, delta_rack, drop, confidence=0.95,
                   min_tail=4, seed=20260917, bootstrap=2000):
    """Risk of deferring a purchase on a WAIT signal.

    The headline CVaR is inherently a small-sample statistic: at 95% even a
    100-observation WAIT sample puts only five points in the tail, and the
    previous implementation quoted a bare number from exactly that, where
    dropping one observation moved it by a third.  Rather than pretend to a
    precision the data does not support, this returns the point estimate
    *together with* a bootstrap interval and the tail size, so every consumer is
    forced to surface the uncertainty.

    ``probability_adverse`` and ``median_move`` are reported alongside because
    they are estimated from the whole WAIT sample and are therefore far more
    stable than the tail mean.  Those are the numbers an operator should lead
    with; the CVaR is the stress case.
    """
    x = np.asarray(delta_nymex, dtype=float)
    y = np.asarray(delta_rack, dtype=float)
    wait_moves = y[x <= drop]
    if wait_moves.size < 20:
        return None
    cutoff = float(np.percentile(wait_moves, confidence * 100))
    tail = wait_moves[wait_moves >= cutoff]
    if tail.size < min_tail:
        return None

    rng = np.random.default_rng(seed)
    draws = rng.choice(wait_moves, size=(bootstrap, wait_moves.size), replace=True)
    cut = np.percentile(draws, confidence * 100, axis=1, keepdims=True)
    boot = np.array([row[row >= c].mean() for row, c in zip(draws, cut.ravel())])
    low, high = np.percentile(boot, [2.5, 97.5])

    return {
        "cvar": float(tail.mean()),
        "cvar_low": float(low),
        "cvar_high": float(high),
        "tail_n": int(tail.size),
        "sample_n": int(wait_moves.size),
        "cutoff": cutoff,
        "probability_adverse": float((wait_moves > 0).mean()),
        "median_move": float(np.median(wait_moves)),
    }
