# Walk-Forward Leakage Design

## Objective

Make every historical threshold and performance result reproducible from only
information available before the corresponding purchasing decision.

## Correct Model

Use an expanding or fixed-length rolling-origin evaluation. For decision session
`T`, the calibration set must end at the most recent fully known rack outcome
before `T`. The test set starts at `T` and must never contribute to threshold
selection, smoothing, percentile calculation, or configuration selection.

For each fold:

1. Define the decision timestamp and the latest eligible rack outcome timestamp.
2. Build a training slice ending at that eligible outcome.
3. Apply a purge interval for the rack lag and any settlement-to-rack reporting
   delay, so an outcome cannot appear in both training and evaluation.
4. Fit all candidate parameter sets using the training slice only.
5. Select the objective once, freeze the selected parameters, and score only the
   following evaluation interval.
6. Carry threshold smoothing forward from the prior fold's frozen state, not
   from a value calculated with future folds.

The fold artifact, rather than the current `metrics_cache.json`, becomes the
source of truth for historical replay.

## Data Migration

Add an append-only `calibration_runs` dataset (CSV or JSON Lines) with:

- `effective_session`, `training_start`, `training_end`, and purge duration.
- Source-history hash and row count.
- Candidate grid/version and objective definition.
- Selected thresholds, lean thresholds, daily standard deviations, lag, and
  rolling-window values for RB and HO.
- Prior smoothing state and resulting smoothing state.
- Code/config version identifiers and generation timestamp.

Existing historical predictions cannot be promoted to genuine point-in-time
evidence because their calibration inputs were not preserved. Retain them as
`backfill` estimates. Generate calibration artifacts prospectively, then use
those artifacts for all new replay and report calculations.

## Deployment Plan

1. Implement a pure fold builder returning a calibration artifact without
   writing files.
2. Add a shadow job that produces artifacts alongside the existing calibration
   path and compares thresholds, without changing live signals.
3. Validate the artifact chain over a fixed historical fixture.
4. Switch live calibration to write and consume the artifact for the next
   session only.
5. Make replay require an artifact for the requested date; report `unknown`
   rather than silently substituting current cache values.

## Regression Risks and Tests

The highest risk is changing live thresholds because the old smoothing state is
not reconstructable exactly. Run the shadow job for at least one complete
calibration window before cutover.

Required tests:

- Mutating any row after a fold's training end cannot alter its artifact.
- A test row never occurs in its own or an earlier fold's training set.
- A row inside the purge interval is excluded from both sides.
- Replaying an artifact yields the same thresholds and signal byte-for-byte.
- Smoothing for fold `N` depends only on artifacts through fold `N - 1`.
