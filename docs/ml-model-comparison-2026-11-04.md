# ML Model Comparison: November 4, 2026

## Decision

Retain the production OLS model for purchasing signals. Huber regression is the
strongest new challenger for gasoline point forecasts and merits a prospective
shadow trial. Its historical MAE improvement survives a correction across six
comparisons and sensitivity to three serial-dependence block lengths. That
does not establish better executable savings or justify an immediate switch.
Diesel does not have a comparably robust improvement. The tested Ridge and
boosted residual models do not earn promotion.

## Data and Controls

Dataset: 881 history rows through November 4, 2026; SHA-256
`599deb8cf1be73a2372ef35940eed4e704fef5241554be4ab91bcd2a7e33b34a`.
All models are scored on the same 413 eligible sessions for each fuel, from
October 7, 2024 through November 4, 2026. The existing alignment, contiguous
session, and calendar roll filters are reused.

Five models are compared:

1. Production OLS family: the 360-observation regression, current empirical
   residual probabilities, and current full/lean threshold rules.
2. Same-adapter OLS: the same point forecast as production, with the identical
   prequential probability calibration and action adapter used for challengers.
3. Huber regression: 360 observations, standardized futures move, epsilon 1.35,
   alpha 0.0001. Huber estimates a robust location, not a guaranteed conditional
   mean or expected monetary return.
4. Ridge residual correction: standardized eight-feature inputs, alpha 10,
   SVD solver. Its output corrects the baseline point forecast.
5. Gradient-boosted residual correction: 60 trees, depth 2, minimum 30 samples
   per leaf, learning rate 0.05, squared-error loss, random seed 1729.

These settings were fixed before the initial comparison. There is no parameter
search over test results. The same-adapter control distinguishes a change in
calibration/policy from a change in point-forecast model. The production control
is a reconstructed model family, not an exact replay of historical immutable
artifacts, calibration activation delays, proxy decisions, or failed executions.

## Availability and Leakage Safeguards

Each forecast is refitted chronologically. Its fit excludes its own outcome,
all future outcomes, and one earlier eligible observation as a purge gap.
Earlier test outcomes may enter later fits once they become available; this
is an online evaluation, not a fixed train/test partition.

Ridge and boosting use these inputs:

- Current futures move and its absolute magnitude.
- The exact previous session's rack-minus-futures basis, in cents per gallon.
- The exact previous session's earlier-only baseline forecast error, plus a
  missing-error indicator when there is no valid forecast for that date.
- Standard deviation of the last 20 earlier eligible futures moves.
- Sine/cosine seasonality terms based on the known calendar date.

Prior-session features join by date. A gap never shifts a stale forecast error
onto a later date. Current rack prices never enter their own feature vector.
The previous rack posting is assumed to have arrived before the next decision;
complete historical email receipt timestamps are unavailable, so actual
point-in-time availability cannot be proven for every historical row.

Residual learning targets are errors from baseline forecasts generated using
only earlier observations. They are not in-sample fitted residuals. The
challengers' probabilities and 80% prediction ranges use the last 120 earlier
out-of-sample errors, also behind the purge boundary. Production retains its
existing probability calculation as a separate control; its experimental
interval uses the common OOS adapter because production has no equivalent
interval field in this research comparison.

The initial baseline warmup is 120 observations; corrections need 80 earlier
forecast-error targets. Scoring begins only after the baseline has 360 fitting
observations and every challenger has 120 prior calibration errors. Candidate
fit failures abort the experiment rather than silently discarding difficult
dates or substituting another model.

## Results

MAE and RMSE are cents per gallon of rack-change prediction error. Lower is
better. Brier evaluates `rack change > 0` on all scored dates, including dates
when a model abstains. Alert precision alone is not used to rank models, since
different models choose different alert sets.

| Fuel | Model | MAE | RMSE | Brier | Alerts | Correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gasoline | Production OLS | 1.841 | 2.984 | 0.1240 | 344 | 89.24% |
| Gasoline | Same-adapter OLS | 1.841 | 2.984 | 0.1250 | 342 | 88.89% |
| Gasoline | Huber | 1.689 | 3.008 | 0.1206 | 355 | 88.45% |
| Gasoline | Ridge correction | 1.983 | 2.949 | 0.1462 | 313 | 86.58% |
| Gasoline | Boosted correction | 1.818 | 2.886 | 0.1300 | 326 | 89.26% |
| Diesel | Production OLS | 1.974 | 3.392 | 0.0827 | 387 | 91.73% |
| Diesel | Same-adapter OLS | 1.974 | 3.392 | 0.0819 | 385 | 91.95% |
| Diesel | Huber | 1.927 | 3.369 | 0.0812 | 385 | 91.95% |
| Diesel | Ridge correction | 2.156 | 3.526 | 0.0818 | 375 | 92.53% |
| Diesel | Boosted correction | 2.089 | 3.403 | 0.0863 | 369 | 92.41% |

Huber reduces full-period MAE by 8.24% for gasoline and 2.39% for diesel.
Gasoline RMSE worsens slightly, indicating that lower typical error does not
eliminate the large-error tradeoff. Its gasoline gross timing-payoff
improvement is only about 0.025 cents per scored session, with an unadjusted
95% paired interval that includes zero. The diesel policy gives identical
actions and gross payoffs to the same-adapter OLS control.

The latest 90 eligible sessions, June 23 through November 4, are reported
separately as a diagnostic slice. They do not constitute an untouched holdout.
Boosting looks more attractive in this slice than in the full period, which
is a reason to avoid selecting it from that slice alone.

## Statistical Inference

The primary loss comparison is full-period paired absolute forecast error.
The family contains three challengers versus production OLS for each of two
fuels: six comparisons. Bonferroni uses a 99.1667% interval for each comparison
to allocate a 5% family error budget. Moving-block bootstrap intervals use
20,000 resamples with common draws across models; block lengths of 5, 10,
and 20 eligible observations are checked for sensitivity.

Huber's gasoline MAE improvement is 0.1518 cents. Corrected intervals are:

| Block Length | Corrected Interval, Cents |
| --- | --- |
| 5 | [0.0328, 0.2671] |
| 10 | [0.0290, 0.2564] |
| 20 | [0.0145, 0.2672] |

The corresponding diesel intervals include zero at every block length.
Ridge and boosting have no positive corrected lower bound for either fuel.

These are approximate bootstrap intervals under temporal dependence, not exact
guarantees for arbitrary regime shifts. The correction covers this declared
six-comparison family, not all historical research, earlier window experiments,
secondary metrics, or an unlimited search. A positive result on this reused
history remains a research finding requiring prospective confirmation.

Secondary metrics include Brier, RMSE, wrong WAIT counts, gross timing payoff,
and prediction-range coverage/width/interval score. Their 95% intervals and all
subperiod intervals are exploratory and unadjusted. No p-value is used to
claim the system is economically profitable.

## Risk and Practical Limits

Experimental nominal 80% ranges cover only about 72%-77% of outcomes for most
full-period candidates, so their nominal coverage should not be trusted for
operational guarantees. The proper interval score penalizes both wide ranges
and misses. This experiment does not deploy these ranges in alerts.

Gross timing payoff is `direction * observed rack change`, with zero for an
abstention. It assumes equal volume and excludes truck availability, inventory,
fees, and the supplier's price-lock policy. It is not actual realized savings.
RB and HO are evaluated separately, preserving their shared-market dependence
rather than counting both commodities as independent market days.

There is no genuinely untouched final holdout in this history: it has already
been inspected and used in prior model development. No model can be certified
as universally best or highly accurate in every future regime from these
results. Production model, signals, cache, calibration ledger, and price
history remain unchanged.

## Prospective Promotion Standard

Freeze Huber as the gasoline shadow challenger before collecting new results.
Record its input availability timestamps, actual contracts and baselines,
point forecast, probabilities, interval, model parameters, and recommendation
alongside production without changing purchasing alerts. Keep error scoring
separate from supplier price-lock and actual execution economics.

Before examining future results, set a review schedule and a minimum useful
benefit based on actual costs. Assess paired forecast error, probability
quality, adverse WAIT outcomes, and net executable value over identical future
dates and comparable load volumes. Avoid repeatedly switching to the current
winner. A model must improve economically relevant behavior and retain stable
risk before promotion; a lower historical MAE alone is insufficient.

## Reproduction

```sh
python3 -m pip install -r requirements-research.txt
python3 scratch/compare_ml_models.py --output-dir reports
python3 -m pytest -q
```

Research dependencies are installed in CI for the regression tests. Scheduled
tracker and ingestion jobs continue to use the production requirements only.
The research CLI refuses output directories outside `reports/`.

Validation: 320 repository tests pass with zero failures, including 12 new ML
causality, exact-date alignment, policy, paired-comparison, and corrected-block
bootstrap checks. Data and dependency source hashes match the saved report;
all 4,130 forecast rows have unique date/commodity/model keys, matching sample
counts, and fitting/calibration boundaries strictly before their test dates.

The JSON report records data, source/dependency code hashes, package versions,
features, hyperparameters, and scoring rules. The companion forecast CSV
contains all 4,130 scored model/commodity/date records and their fitting and
calibration boundaries. These files are research artifacts, not live predictions.

- `reports/ml_model_comparison_2026-11-04.json`
- `reports/ml_model_comparison_2026-11-04_forecasts.csv`
- `scratch/compare_ml_models.py`
- `test_ml_model_comparison.py`

Model implementations use the official scikit-learn
[Ridge](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html),
[Huber](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.HuberRegressor.html), and
[gradient boosting](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
estimators. Cross-validation, forecast distribution construction, and
purchasing-policy evaluation are explicit research code with causality tests.
