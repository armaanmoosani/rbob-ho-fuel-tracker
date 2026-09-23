# Quantitative purchasing policy

## Production model

The production forecast remains the point-in-time affine pass-through model:

`rack_change = intercept + slope * NYMEX_change + residual`

Direction probabilities use the empirical residual distribution. The 75%
probability boundary and settlement-snapshot noise floor remain risk
guardrails. They are not interpreted as an economic optimum.

An action must now also have positive expected economic value:

- BUY: `expected rack rise - incremental dispatch/carry cost`
- WAIT: `expected rack decline - deferral/stockout/expedite cost`

The four commodity/action costs live only in `data/config.json`. Their defaults
are zero because the repository cannot infer physical operating costs. At zero,
the new gate preserves the existing signal policy and the alert states that its
net value excludes unconfigured costs. Enter costs in cents per gallon:

- `RB_BUY_INCREMENTAL_COST_CENTS_PER_GAL`
- `RB_WAIT_INCREMENTAL_COST_CENTS_PER_GAL`
- `HO_BUY_INCREMENTAL_COST_CENTS_PER_GAL`
- `HO_WAIT_INCREMENTAL_COST_CENTS_PER_GAL`

These keys are intentionally excluded from `metrics_cache.json`; a generated
cache must never override a later operator policy edit.

## Calibration uncertainty

Each nightly calibration now performs a deterministic five-session
moving-block bootstrap. It refits the slope and empirical residual distribution
on every draw and publishes 95% intervals for both action thresholds. Live
alerts report whether the observed NYMEX move clears the conservative edge of
that interval. This is parameter stability, not an interval for tonight's rack
price.

If an immutable artifact already occupies the next session, a new method is
scheduled on the next unused business session. Until then the tracker activates
the exact artifact for the current session and removes every future-only model
field before applying it. Operator cost inputs are preserved separately.

## Live evaluation

Before 30 live observations the weekly report still withholds a permutation
p-value, but it now reports:

- correct/total and a Beta(1,1) posterior 95% interval;
- average savings and worst adverse result;
- Brier score from probabilities captured at decision time;
- separate RB and HO rows;
- the selective policy versus following the NYMEX sign every scored day;
- a paired moving-block interval for incremental policy savings.

Actual execution is separate from modeled opportunity. Use
`python record_execution.py --help` to record gallons and, when available, the
paid and alternative prices. Realized dollars are calculated only from those
two prices.

## Features deliberately not promoted

On 601 strictly later observations per commodity, using a 360-row rolling
training window, the following candidate comparison was run before this change:

| Model | RB Brier | RB precision | HO Brier | HO precision |
|---|---:|---:|---:|---:|
| Pooled empirical residual benchmark | 0.11368 | 92.9% | 0.07477 | 93.3% |
| Calm/volatile residual split | 0.11257 | 91.2% | 0.07567 | 93.0% |
| Lagged basis, basis change, weekday ridge | 0.12273 | 91.9% | 0.07877 | 93.5% |

The volatility split's tiny RB Brier improvement came with lower precision and
average savings and did not repeat for HO. The basis model worsened probability
calibration for both products. Neither is in production. Full GARCH, neural
nets, crude-oil overlays, and broad technical indicators are likewise omitted
until a predeclared nested walk-forward comparison shows repeatable incremental
decision value after costs.
