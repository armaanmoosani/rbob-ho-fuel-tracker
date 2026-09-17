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

## The repair

`migrate_legacy_alignment.py` performed the correction on 2026-09-17.

**The offset is one calendar day, not one business day.** Legacy weeks are
stamped Tue..Sat and never Monday, which is only consistent with a fixed
calendar shift. Using business days collides at holidays: Good Friday 2023 sent
both the Friday and the Saturday row to Thursday 2023-04-06.

**Settles came from inside the file wherever possible.** For a row stamped `D`
moving to session `S = D - 1 day`, the row already stamped `S` holds exactly
that session's settle, so 758 of the corrections are a byte-for-byte move of an
existing string. Only 244 Monday settles needed an external source, because the
legacy stamping never produced a Monday row and those settles were consequently
never backfilled. Two corrected sessions fall on market holidays and correctly
carry no settle.

**Which rows moved.** 502 of 846. Weeks were classified by their day-of-week
pattern: a Saturday and no Monday means legacy, a Monday and no Saturday means
live, both means a changeover week. Weeks showing neither were treated as legacy
only when they predate the very first live-stamped week, since the live
convention did not exist yet; that promotion was worth 27 rows. The seven
changeover weeks were left untouched — rows inside them cannot be classified
with confidence, and guessing would inject the very error this removed.

### Gates

Nothing was written until all eight passed:

| Gate | Result |
|---|---|
| A External source reproduces every legacy settle in the file | 918 checked, max deviation **0.000000** |
| B Internally-sourced settle agrees with the external one | 758 cross-checked, max deviation **0.000000** |
| C No two rows land on the same date | 846 rows, 846 distinct dates |
| D Rack values conserved as a multiset | all 846 triples preserved exactly |
| E No row created or destroyed | 846 → 846 |
| F Rows in the already-verified era untouched | 269 rows byte-identical |
| G The correction actually fixes the alignment | see below |
| H Rewritten file passes `validate_graves_history` | accepted |

Gate A is the one that made the rest trustworthy: the external series reproduces
every settle already in the legacy portion of the file to zero deviation, which
simultaneously validates the source and confirms that the file's settles are
keyed to the stamped date. (In the *live* era the same comparison disagrees on
about 25% of rows, because the live ingest captures contract-specific Schwab
settles while the external series is continuous front-month. That era was not
touched.)

### Result

| | before | after |
|---|---|---|
| RB lag-1 coefficient | +0.157 (p < 1e-7) | **+0.013 (p = 0.44)** |
| HO lag-1 coefficient | +0.095 (p = 6e-5) | **+0.003 (p = 0.81)** |
| RB pass-through slope | 0.516 | 0.634 |
| HO pass-through slope | 0.874 | 0.955 |
| Usable pairs | 235 | **728** |
| Legacy-stamped weeks | 109 | 1 |

The slopes now match what the already-trusted era showed on its own (0.619 and
0.953), which is the strongest confirmation that the re-dating is correct rather
than merely tidier.

Calibration consequences:

* `CALIBRATION_ERA_START` moved to the first session, 2023-03-06.
* `ROLLING_WINDOW_DAYS` rose 180 → 360; lengthening the fit window was the only
  change that improved out-of-sample calibration, and it only became possible
  with the extra history.
* The walk-forward evaluation widened to six 45-session blocks, so the published
  out-of-sample figure spans ~16 months across both a calm and a volatile
  regime instead of only the recent spike.
* **Measured precision fell** — RB 96.0% → 93.5%, ¢/alert 6.93 → 4.12. That is
  the estimate becoming honest, not the model getting worse.

## The archived original

`data/graves_history.pre_alignment_migration.csv` is the file as it stood before
the migration. It is retained deliberately: the live file is now aligned, so the
archive is the only remaining positive test case for the alignment detector.
`test_alignment.py::test_the_archived_pre_migration_file_is_still_detected_as_misaligned`
and the `verify_statistics.py` check of the same name both assert it is still
flagged. **Do not delete it** — without it, a broken detector would pass
silently.

The integrity registry was re-baselined by appending a new record for
`graves_history.csv`. The pre-migration hash remains in the registry as the
record of what the data was, and the evidence that the change was authorised
sits beside it in git: the archived CSV, the migration script, and the commit
carrying both.

## What is still not corrected

The seven changeover weeks listed by `alignment.convention_weeks(...)
["interleaved"]`. They are ~30 rows out of 846 and the aggregate lag-1 test
passes comfortably with them included, so they are left in place rather than
guessed at.

## Verification

```bash
python3 alignment.py                            # current state
python3 alignment.py --scan                     # lag-1 by candidate era start
python3 migrate_legacy_alignment.py             # dry run; re-checks every gate
python3 -m pytest test_alignment.py -q
```
