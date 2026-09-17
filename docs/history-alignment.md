# History alignment

## The defect

`data/graves_history.csv` contains rows written by two different producers,
using two incompatible conventions for what the `date` column means.

**Live ingest (`ingest_prices.py`)** stamps a row with the session date `D`:
`nymex_*` is the 1:30 PM CT settle of `D`, and `rack_*` is the price Graves
posted the evening of `D`. This is the convention the model assumes
(`LAG_DAYS = 0`).

**Bulk import (`historical_import/extract_emails_to_csv.py`)** stamped rows from
the email `Date:` header, formatted with `strftime("%Y-%m-%d")` *without first
converting to America/Chicago*. Graves sends the rack email around 8 PM CT. A
header carrying a UTC offset puts that at 01:00 the following day, so the row
landed one calendar day late — and its rack price was then joined against the
**next** day's settle.

## Evidence

Fit `rack_delta_t = a + b0·nymex_delta_t + b1·nymex_delta_{t-1}`. A
same-session series loads onto `b0`; a series stamped one day late loads onto
`b1`.

| Window | RB b0 | RB b1 | HO b0 | HO b1 |
|---|---|---|---|---|
| Full file | 0.516 | **+0.157** (p < 1e-7) | 0.874 | **+0.095** (p = 6e-5) |
| Verified era | 0.619 | −0.015 (p = 0.56) | 0.953 | −0.008 (p = 0.64) |

The day-of-week histogram is the same defect seen structurally. The legacy
stamping shifts every row forward one day, which erases Mondays and invents
Saturdays:

| Year | Monday rows | Saturday rows |
|---|---|---|
| 2023 | 0 | 42 |
| 2024 | 2 | 45 |
| 2025 | 24 | 20 |
| 2026 | 31 | 2 |

Consequences that were visible in the output before the fix:

- Yearly R² of the pass-through regression read 0.085 / 0.050 / 0.132 / 0.715
  for 2023–2026. The README attributed that fourteen-fold jump to a change in
  market regime. It was the convention changing over.
- Roughly 60% of the training pairs matched a rack price to the wrong settle,
  attenuating the fitted slope and depressing every historical precision
  figure.

## Why the legacy rows are not repaired

Shifting a legacy row back one session requires the settle for that earlier
session. Under the legacy stamping **no row was ever stamped Monday**, so
Monday settles were never backfilled and are simply absent from the file. Only
Wednesday, Thursday and Friday sessions could be reconstructed, which would
leave a systematic weekday hole in the training set.

Worse, the changeover was not clean. Seven weeks contain rows from both
conventions — a Monday row and a Saturday row in the same week — the last being
`2025-07-28/2025-08-03`. Individual rows inside those weeks cannot be
classified with confidence, and guessing wrong injects a one-session
misalignment directly into the live thresholds.

## What was done instead

`CALIBRATION_ERA_START = 2025-08-04`, the first Monday strictly after the final
interleaved week. It satisfies two independent criteria:

1. **Statistical** — the lag-1 coefficient is insignificant for both
   commodities from here on. On its own this criterion would permit a start as
   early as 2024-05.
2. **Structural** — no week after the cut contains both a Saturday and a Monday
   row.

The legacy rows stay in the file. They are excluded from calibration, from the
walk-forward evaluation, and from every published statistic.

Two further filters do the rest of the work:

- `alignment.aligned_deltas` keeps a pair only when the previous surviving row
  is the immediately preceding NYMEX business day. Without it, a missing
  session (a failed ingest, or one of the 121 rows with no settle) makes its
  neighbours adjacent and silently turns their difference into a *two*-session
  move scored against one-session thresholds.
- `alignment.assert_calibration_alignment` runs before every calibration and
  aborts if the verified window develops a lag-1 pass-through, or if
  legacy-stamped weeks exceed 15% of the era. Isolated late-stamped Fridays are
  expected and harmless — the weekend filter and the contiguity filter discard
  the affected pairs — so the structural gate measures a rate, not a single
  occurrence.

## Recovering the legacy era

Possible, and worth doing: it would roughly triple the calibration sample and
let the model be validated across a calm regime as well as a volatile one.

1. Backfill Monday NYMEX settles for 2023-03 through 2025-07 from an external
   source (yfinance `RB=F` / `HO=F`, or Schwab price history).
2. For each legacy row, set its session date to the previous NYMEX business day
   of its current stamp, and re-join the settle for that corrected date.
3. Leave the seven interleaved weeks out; they cannot be classified.
4. Re-run `python3 alignment.py --scan`. If `b1` is insignificant across the
   whole file, move `CALIBRATION_ERA_START` back and regenerate the README with
   `python3 generate_readme_stats.py --write`.

Do not skip step 4. The alignment gate is the only thing standing between a
stamping error and the live thresholds.

## Verification

```bash
python3 alignment.py          # current state of both eras
python3 alignment.py --scan   # lag-1 coefficient by candidate start date
python3 -m pytest test_alignment.py -q
```
