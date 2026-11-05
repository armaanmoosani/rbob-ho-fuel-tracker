# Prospective Gasoline Huber Trial v1

## Registered Design

The trial starts with the November 5, 2026 decision window. Its registered
manifest is `data/shadow_trial_plan.json`, created before the first prospective
verdict, with a content hash and frozen implementation/dependency hashes.
Gasoline only is included. No shadow forecast changes a purchasing alert,
production prediction row, cache, calibration artifact, or price history.

The challenger is Huber regression with a 360-eligible-observation fit window,
one-observation purge, epsilon 1.35, alpha 0.0001, tolerance 1e-6, maximum 1,000
iterations, standardized futures moves, and scikit-learn 1.7.1. Its probability
calibration uses 120 earlier out-of-sample errors behind the same purge gap.
Full/lean confidence thresholds are 0.75/0.65 with a 0.6-cent futures-move floor.
No setting is selected from future trial outcomes. A source change to frozen
components stops the shadow trial for review; it cannot silently change the
experimental treatment.

## Capture and Timing

After the ordinary daily verdict is sent or attempted, the primary tracker
freezes a separate input during 14:35-15:59 CT. Only one input may exist per
session. Retries use that exact input; there is no reconstruction from later
history. No capture occurs before the trial start, outside the verdict window,
on weekends, or during the tracker's holiday suppression.

Eligibility is determined before the outcome and recorded even if rejected:

- Exactly one live RB production row from the verdict window.
- A Schwab settlement-window snapshot from 13:30-13:45 CT, with matching
  signal/baseline contract identifiers and verified provenance.
- A calibration-matched prior settlement and exact prior-session rack posting.
- The same runtime config hash as the immutable production prediction.
- No current or future rack posting already present at capture.
- At least 482 earlier eligible historical pairs, allowing the complete frozen
  360-row fit and 120-error prequential calibration with their purge boundaries.

Each sealed JSON input includes those 482 pairs, source history hash, exact
production prediction/parameters, prior rack price, contract identifiers,
timestamps, and run identifier. The outcome field is explicitly omitted from
the copied production row. The model worker reads this sealed payload only,
even if a later outcome has since been ingested.

The worker is a separate Actions job with its own concurrency group and ML
dependencies. It cannot hold the primary tracker's job lock. Both start and
completion must respect the frozen 16:00 CT cutoff to qualify as an on-time
shadow forecast. Missing inputs cannot be invented by dispatching the worker
later; an existing sealed input is mandatory.

## Storage and Integrity

- `data/shadow/inputs/YYYY-MM-DD.json`: immutable availability/eligibility record.
- `data/shadow/forecasts/YYYY-MM-DD.json`: on-time forecast or explicit failure,
  late completion, or ineligible status.
- `data/shadow/outcomes/YYYY-MM-DD.json`: independently appended actual outcome
  and matched production/challenger scores.
- `data/shadow/reviews/primary.json`: a single fixed-cohort primary review.

Files use atomic exclusive creation and canonical content hashes. Input,
forecast, outcome, and review hashes link the records. Validation also checks
time boundaries and recomputes outcome statistics; a valid JSON hash alone does
not establish mathematical consistency. These checks and the git history
provide reproducibility and tamper detection, not third-party cryptographic
attestation. Existing forecasts and outcomes cannot be replaced by retries.

The core tracker captures inputs without importing scikit-learn. A shadow
capture error is reported as a warning after the primary verdict path. A model
failure is recorded in the isolated ledger. Shadow integrity errors fail only
the isolated worker, and the weekly report withholds shadow results if its
audit fails. Primary production validation remains independent.

## Outcomes, Failures, and Gaps

Each completed ingestion calls the isolated reconciler, including runs that
find no new missing price date. The reconciler checks all sealed pending inputs,
so an email arriving the following morning still scores its actual effective
date. It joins the current date directly and compares to the prior rack price
frozen before the decision; the final CSV row is scoreable and weekends do not
shift the outcome. A changed/missing prior posting or changed already-scored
outcome raises an integrity error.

The main comparison is intention-to-treat: if the challenger fails or misses
its deadline, the production forecast/action is retained for that session in
the comparison, producing zero incremental benefit. The failure stays visible
in coverage counts. Successful-only results must not be substituted for the
main comparison. Completed expected tracker sessions with no capture are
listed explicitly and block a clean promotion-quality result.

The primary metric is the paired reduction in absolute rack-change forecast
error, in cents per gallon. Secondary measures are RMSE, Brier for `rack change
> 0`, wrong-WAIT costs, and gross timing proxies. Zero price moves do not count
as correct directional calls. A timing proxy is not realized savings; no dollar
benefit is manufactured from it. Actual executions remain in the independent
execution ledger and require verified paid/counterfactual prices.

Huber's output is a robust location estimate, not a guaranteed conditional mean
or monetary expected value. Its empirical nominal 80% ranges are experimental
and explicitly not operationally validated. Neither those ranges nor its
probabilities are exposed as replacement purchasing guidance.

## Fixed Review and Power

After 60 resolved pairs, the operational review remains descriptive: data
coverage, timing, failures, and stability. No significance decision is made.

The single formal review uses the first 320 eligible captured sessions in
chronological order. Every member of that cohort must be resolved before
inference unlocks; later results cannot replace an earlier pending member.
The review is written once and is not recomputed when later data arrives.

Planning uses historical paired-loss variance 0.87596 cents squared, a target
improvement of 0.15 cents per gallon, one-sided alpha 0.025, and approximately
80% power under a stable normal-approximation planning model. The resulting
requirement is about 306 paired observations, rounded to 320. That is roughly
15-18 months of eligible sessions, potentially longer after exclusions and
outages. The planning assumptions may fail under a regime change; sample size
does not guarantee significance or a profitable policy. Detecting a smaller
0.10-cent effect at comparable planning power would need about 688 pairs.

The formal analysis uses paired moving-block bootstrap intervals with 20,000
resamples and sensitivity to block lengths of 5, 10, and 20. All lower bounds
must exceed zero, and the observed mean improvement must be at least 0.10 cents
to pass the forecast-effect gate. The lower bound being positive establishes
evidence relative to zero under the approximation; it does not establish that
the true effect exceeds 0.10 cents. On-time forecast coverage must be at least
95%, with no uncaptured completed expected session. There is only one fuel and
one preselected primary challenger; secondary metrics are exploratory.

No repeated early significance checks, outcome-driven parameter tuning,
replacement of pending dates, or automatic promotion are allowed. Even passing
the forecast gates requires separate adverse-risk and executable-purchasing
review with actual execution evidence before any live model change.

## Operation and Verification

```sh
python3 shadow_trial.py status
python3 shadow_trial.py reconcile
python3 -m pip install -r requirements-shadow.txt
python3 shadow_trial.py verify-model
python3 -m pytest -q
```

Publishing shadow records is restricted to GitHub Actions. The reusable shadow
workflow is called after a captured tracker verdict and after successful daily
ingestion. It may also be dispatched manually to verify runtime or reconcile
existing records; this cannot fabricate a prospective input. The weekly report
contains a separate shadow section and never includes these rows in production
precision, live counts, or significance tests.

Tests cover prospective eligibility, live-record preservation, exact model
parity with research, purged data, immutable retries, outcome changes, delayed
emails, late workers, failure accounting, missing sessions, fixed-cohort
selection, protocol changes, and publishing guards. No synthetic test data is
written into the live trial ledger.
