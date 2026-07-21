# Conviction Forward-Contamination Design

## Objective

Make historical conviction labels reflect the values available at the decision
time, rather than a later `metrics_cache.json` value.

## Correct Model

Conviction is a decision-time feature. Each live prediction must persist the
exact denominator and result used to make it:

- `nymex_daily_std_used`, `z_score`, and `conviction`.
- Hike, drop, and lean thresholds used.
- Calibration effective session, calibration artifact identifier, and config
  hash.
- Signal price, baseline price, and both settlement provenance identifiers.

Reports and validation should group historical rows using these persisted
values. They must not recompute a historical Z-score from today's configuration
or use the current metrics cache as a proxy for the past.

## Data Migration

Append the fields above to `prediction_log.csv`. Rows created before this
migration receive `unknown` provenance and no reconstructed conviction label.
Backfill scripts may calculate a separately named historical estimate only when
they have a matching point-in-time calibration artifact; otherwise they must
leave conviction unknown.

Add a log schema version and keep the writer append-only. Validation should
require complete conviction provenance for new `live` rows while accepting
explicit `unknown` only for pre-migration or `unlabelled` rows.

## Deployment Plan

1. Extend the live signal return value to expose all decision-time inputs.
2. Write the new fields for live rows while retaining existing columns.
3. Add report views that show `captured`, `estimated`, and `unknown` conviction
   samples separately.
4. Switch historical conviction analysis to captured values only.
5. After sufficient live observations, deprecate analyses based on current
   config reconstruction.

## Regression Risks and Tests

The main risk is schema compatibility with existing CSV readers and GitHub
Actions state size. Add columns once, make parsers tolerant of legacy rows, and
version the schema before requiring the new fields.

Required tests:

- Changing `metrics_cache.json` after a prediction does not change its stored
  Z-score or conviction in reports.
- A newly written live row contains all required decision-time fields.
- A legacy row is visibly `unknown`, never silently recomputed as live evidence.
- Backfilled rows cannot use a future calibration artifact.
- Conviction-bin precision excludes rows without captured conviction provenance.
