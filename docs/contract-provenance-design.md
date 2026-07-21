# Contract Provenance Design

## Objective

Ensure each live signal compares prices from the same futures contract whose
settlement is recorded for later rack and performance analysis.

## Correct Model

Treat a settlement as a contract-specific observation, not merely a commodity
price. The active Schwab symbol chosen for the 1:30 PM CT decision must be
persisted with the settlement and copied into the prediction record. The next
day comparison must either use the same contract or explicitly identify a roll
and suppress or normalize the cross-contract comparison.

The data path should be:

1. Resolve the active Schwab symbol once for the CME session.
2. Capture the settlement-window price with that symbol, source timestamp, and
   source identifier.
3. Store the same provenance with the live prediction and its baseline
   settlement.
4. When evaluating the outcome, join by session and commodity while retaining
   the recorded contract identifiers; never infer them from today's front month.

## Data Migration

Extend `graves_history.csv` or introduce a normalized `nymex_settlements`
dataset with, at minimum:

- `session_date`, `commodity`, `price`, `schwab_symbol`, and Yahoo symbol.
- `source`, `captured_at`, and the settlement-window timestamp.
- `previous_symbol`, `roll_flag`, and a provenance schema version.

Add `schwab_symbol`, `baseline_schwab_symbol`, and settlement identifiers to
new `prediction_log.csv` rows. Existing rows lack reliable contract identity;
mark their provenance `unknown` and keep them out of contract-sensitive
validation. Do not fabricate symbols from current rollover rules.

## Deployment Plan

1. Write a contract-aware settlement record in parallel with the current daily
   settlement file.
2. Add read-only diagnostics that flag a prediction whose baseline and current
   symbols differ without a recorded roll suppression.
3. Start recording provenance on live predictions and reports.
4. Make ingestion populate the normalized settlement record, then migrate live
   signal construction to read it.
5. Only after several roll cycles, retire contract inference from historical
   replay.

## Regression Risks and Tests

The primary risk is an operational gap at the 1:30 PM snapshot that leaves no
price available. Preserve the existing fallback behavior, but label fallback
provenance and suppress a decision if its contract identity cannot be matched.

Required tests:

- A normal session persists identical contract IDs in settlement and signal.
- A calendar roll with an unchanged resolved symbol does not create a false
  cross-contract gap.
- A true roll suppresses or explicitly handles the comparison.
- Sunday-open and holiday sessions resolve to the correct CME session date.
- Historical `unknown` provenance cannot be counted as verified live evidence.
