# Pass-Through Model Comparison: November 4, 2026

## Recommendation

Keep the current 360-observation production model. Neither fixed alternative
provides a consistent improvement in prediction error, probability quality,
and gross purchasing-timing payoffs across both the full and recent samples.
Recent-weighted gasoline forecasts merit a future shadow trial for predicting
the size of a rack move; they have not earned a production policy change.

## Experiment

Data: `data/graves_history.csv` through November 4, 2026, SHA-256
`599deb8cf1be73a2372ef35940eed4e704fef5241554be4ab91bcd2a7e33b34a`.

The experiment uses the existing alignment checks, contiguous-session deltas,
and roll exclusions. Each candidate predicts the same 413 eligible sessions per
commodity, October 7, 2024 through November 4, 2026. Every forecast is refitted
using only earlier data, with a one-observation purge before its test outcome.
Earlier test outcomes may enter later fits after that gap, as they would become
available operationally. No forecast uses its own outcome or a later outcome.

Three fixed candidates are compared without a parameter search:

- Current: ordinary least squares over 360 eligible observations.
- Short: the same fit over 180 eligible observations.
- Recent-weighted: weighted least squares over 360 eligible observations,
  with a coefficient-weight half-life of 90 eligible observations. Empirical
  residual calibration remains unweighted over that training window, isolating
  the effect of changing the coefficient fit.

All candidates use the existing full/lean confidence targets and 0.6-cent
calibration-matched baseline floor. Lean and full alerts both enter the action
metrics. Point errors and the Brier score for the event `rack change > 0` are
measured on all test rows, including rows where a candidate abstains. Alert
precision excludes abstentions and counts zero rack moves as incorrect.

This is a comparison of reconstructed model families, not a replay of the
historical production artifacts, their delays, or fallback decisions. It is
not a new live track record. This history has already informed model
development, so even the recent slice is not an untouched final holdout.

## Results

Mean absolute error (MAE) is in cents per gallon. Lower MAE and Brier are better.

| Fuel | Model | Full MAE | Full Brier | Alerts | Correct |
| --- | --- | ---: | ---: | ---: | ---: |
| Gasoline | Current 360 | 1.841 | 0.1240 | 344 | 89.24% |
| Gasoline | Short 180 | 1.876 | 0.1296 | 325 | 89.23% |
| Gasoline | Recent-weighted 360 | 1.838 | 0.1262 | 330 | 89.39% |
| Diesel | Current 360 | 1.974 | 0.0827 | 387 | 91.73% |
| Diesel | Short 180 | 2.014 | 0.0828 | 386 | 91.97% |
| Diesel | Recent-weighted 360 | 2.006 | 0.0830 | 387 | 91.73% |

The latest 90 eligible sessions run from June 23 through November 4, 2026:

| Fuel | Current MAE | Short MAE | Recent-weighted MAE |
| --- | ---: | ---: | ---: |
| Gasoline | 1.917 | 1.799 | 1.811 |
| Diesel | 2.247 | 2.217 | 2.225 |

Recent-weighted gasoline reduces recent MAE by about 5.5%, but gives exactly
the same actions and gross timing payoffs as the baseline in those 90 sessions.
Over all 413 sessions it has slightly worse Brier and lower gross timing payoff.
The shorter gasoline window performs worse on full-period MAE and Brier.

Paired moving-block bootstrap intervals use 2,000 resamples and blocks of 10
eligible observations. The full-period 95% intervals for the reduction in mean
squared error include zero for both alternatives and both fuels. Recent
gasoline squared-error improvements have positive exploratory intervals, but
the diesel improvements do not. These intervals are not adjusted for the
multiple model/period/metric comparisons and must not justify a significance
claim or automatic promotion.

Gross timing payoff is `sign(action) * observed rack change`, with zero for
abstention. It is an equal-volume opportunity proxy, not actual savings: it
excludes inventory constraints, order deadlines, fees, and actual executions.
No fleet dollar claim is derived from it. RB and HO are analyzed separately;
their shared market shocks are not counted as independent evidence.

## Next Decision

Do not change production coefficients, windows, confidence thresholds, or
calibration artifacts based on this experiment. First record actual purchase
economics and automate daily outcome reconciliation. A subsequent approved
shadow trial should freeze a candidate in advance, archive its decision-time
forecasts, and compare future errors and net executable savings against the
current model over the same dates. Evaluate uncertainty in paired session
blocks; do not repeatedly promote whichever variant currently leads.

## Reproduction

```sh
python3 scratch/compare_adaptive_models.py
python3 -m pytest -q test_model_comparison.py
```

The script writes JSON to standard output and alignment diagnostics to standard
error. It does not edit production data, config, cache, artifacts, or signals.
Full metrics are recorded in `reports/model_comparison_2026-11-04.json`.

Validation: all 308 repository tests pass with the pinned project dependencies,
including seven new causality, purge, adaptation, and paired-date checks.
Repeating the experiment in the default Python environment agrees to within
`1.1e-14` across all numeric fields. Production data and configuration are
unchanged.
