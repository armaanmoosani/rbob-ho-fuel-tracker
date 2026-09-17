# Quantitative Audit — RBOB/HO Fuel Tracker

Scope: logic, accuracy, math, statistics. All numbers below were reproduced against
the repository as of commit `73a7209` (history through 2026-09-16).

---

## Verdict

There is a real, statistically significant signal here. The live sample (75 alerts,
2026-05-21 → 2026-09-11) is 93.3% precise and passes the repo's own date-block
permutation null at **p = 0.0002**. That part holds up.

Almost everything built on top of it does not. The multi-year performance table is
computed over a history that silently mixes **two incompatible date conventions**; the
advertised contract-roll guard **never fires**; the live signal differs from the signal
the model was calibrated on by an amount **comparable to the decision thresholds**; and
four of the eight checks in `verify_statistics.py` **cannot fail by construction**.

The README's headline numbers are not reproducible from this repository.

---

## 1. CRITICAL — `graves_history.csv` mixes two date conventions, offset by one session

This is the single most consequential defect. It invalidates every multi-year statistic
in the project.

### Evidence

Regressing daily rack change on daily NYMEX change, split at 2025-07:

| Era | Series | lag 0 | lag 1 |
|---|---|---|---|
| 2023-03 → 2025-06 | RB | +0.263 | **+0.600** |
| 2023-03 → 2025-06 | HO | +0.314 | **+0.700** |
| 2025-07 → 2026-09 | RB | **+0.822** | −0.033 |
| 2025-07 → 2026-09 | HO | **+0.942** | −0.074 |

In the older half the rack responds to **yesterday's** settle. In the newer half it
responds to **today's**. A rolling 100-row correlation shows the crossover drifting
through 2025 as the two conventions mix.

The day-of-week histogram is the fingerprint:

| Year | Mon | Sat |
|---|---|---|
| 2023 | 0 | 42 |
| 2024 | 2 | 45 |
| 2025 | 24 | 20 |
| 2026 | 31 | 2 |

Pre-2025 there are **no Mondays and 42–45 Saturdays** — every row is stamped one day
late. All 109 Saturday rows carry `NaN` NYMEX (no weekend settle exists to join
against), which is where 109 of the 121 missing NYMEX values come from.

### Root cause

`historical_import/extract_emails_to_csv.py:33` — `parse_email_date()` formats the
email `Date:` header with `strftime("%Y-%m-%d")` **without converting to
America/Chicago**. An 8 PM CT email whose header carries a UTC offset is 01:00 the next
day, so it lands on the following calendar date. The live path
(`ingest_prices.py:696`) stamps rows with the body date instead. The two paths disagree
by exactly one session.

### Impact

- `LAG_DAYS = 0` is correct for the recent third of the history and wrong for the rest.
- On the misaligned rows the "predictor" is a settle that was printed **after** the rack
  it supposedly predicts. That destroys signal rather than manufacturing it — so the
  2023/2024 results are *deflated*, not inflated. Either way they are not measurements
  of this model.
- `verify_statistics.py` yearly R² : **2023 = 0.085, 2024 = 0.050, 2025 = 0.132,
  2026 = 0.715.** The README reads this 14× jump as a market-regime story. It is a
  data-alignment artifact.
- The basis series `rack − nymex` (used by the Mann-Kendall drift monitor) carries a
  spurious one-day component over the same rows.

### Fix

Re-derive the legacy rows from the email bodies' own stated effective date, or shift
them back one NYMEX session and re-join NYMEX. Until then, restrict every published
statistic to the post-convention-change window and say so.

---

## 2. CRITICAL — the contract-roll exclusion never fires

`futures_util.is_contract_roll_day()` returns `False` for **every date of 2026, for RB,
HO and CL**. Verified by enumeration.

```
RB: 0 roll days in 2026    HO: 0 roll days in 2026    CL: 0 roll days in 2026
```

### Mechanism

`get_front_month_contract(dt, prefix, early_roll_days=3)` advances to the next contract
three business days *before* the LTD. `is_contract_roll_day` then tests `today == ltd` —
but by that point `ltd` is already **next** month's LTD, so it can never match. The
`prev_day == p_ltd` fallback fails identically.

```
2026-05-29 (the actual RB LTD):  early_roll_days=3 -> ltd resolves to 2026-06-30
                                 early_roll_days=0 -> ltd resolves to 2026-05-29
```

### This is a proven regression

`scratch/roll_days_baseline.json` — the repo's own pre-refactor snapshot — records
`"2026-04-30": {"RB": true, "HO": true}`. Running `scratch/verify_roll_days.py` today:

```
Mismatches found!
Date 2026-04-30 mismatch with baseline: old RB/HO={'RB': True, 'HO': True}, current=(False, False)
Date 2026-05-01 mismatch with baseline: old RB/HO={'RB': True, 'HO': True}, current=(False, False)
```

The guard that would have caught this lives in `scratch/`, which `pytest.ini` excludes
via `norecursedirs`. It has never run in CI.

### Impact

Every roll-day exclusion in the system is dead: calibration (`backtest.py:240`), OOS
fold scoring, conviction-bin scoring (`backtest.py:395`), CVaR, and the live alert
suppression in `main.py:701`. Related: the README describes the roll as "the 25th of the
month, or nearest business day" — that is the *crude* convention. RB and HO expire on
the **last business day of the preceding month**, which is what the code actually
computes.

---

## 3. CRITICAL — the live signal is not the signal the model was calibrated on

Thresholds are fitted on official **settle-to-settle** changes. Live decisions are made
at 2:35 PM on a snapshot. Comparing the 82 live log rows against the settles later
recorded in `graves_history.csv`:

| | n | mean error | mean abs error | max abs error |
|---|---|---|---|---|
| RB | 42 | +1.66¢ | **1.96¢** | 39.33¢ |
| HO | 40 | +0.76¢ | **1.18¢** | 13.92¢ |

Against live thresholds of **hike +1.70¢ / drop −0.66¢ / lean ±0.50¢**, a mean absolute
error of 1.58¢ is the same size as the decision boundaries.

**10 of 82 live decisions (12%) would have been classified differently** had the official
settle been used. 28 of 82 sit within one mean-error of a boundary.

The two worst discrepancies land inside contract-roll windows — exactly what §2 was
supposed to suppress:

| date | live move | settle move | error | days to RB LTD |
|---|---|---|---|---|
| 2026-08-28 RB | +5.53¢ | −33.80¢ | 39.33¢ | 3 (inside early-roll window) |
| 2026-07-30 RB | −9.75¢ | −27.27¢ | 17.52¢ | 1 |

The error is also biased (+1.66¢ for RB), not zero-mean.

Separately: the LEAN band triggers at ±0.50¢, **3.2× smaller than the mean snapshot
error**. Its measured edge on clean settle data is real but tiny (RB 59.5%, HO 66.4%,
≈ +0.40¢/alert ≈ $34/truck); as actually fired live it is mostly noise.

---

## 4. Statistical validation — four of eight checks cannot fail

`verify_statistics.py`:

| § | Check | Problem |
|---|---|---|
| 2 | "Null model test" | Passes on `mean(null_r2s) < 0.01`. Shuffled R² is always ≈ 1/n ≈ 0.0014. **Tautology.** |
| 5 | "Residual diagnostics" | Passes on `abs(mean_res) < 1e-5`. OLS residuals with an intercept have mean exactly 0. **Tautology.** |
| 4 | "Sensitivity analysis" | Thresholds fitted on the full history, evaluated on the full history. 100% in-sample; reported as "robust". |
| 6 | "Shadow benchmarks" | `np.random.choice(['buy','wait','skip'])` is a uniform 3-way draw, not the "Random 50/50" it is labelled. Its expectation is ~0, so "model beats random" reduces to "savings > 0". Also in-sample, and no variance/p-value. |

Also in that file:
- §1 "Frozen holdout" reads `RB_opt_Hp`/`RB_opt_Dp` from the cache — hyperparameters
  chosen by a **full-history** grid search — so the holdout is not frozen.
- `p_val = np.mean(perm >= real)` with no `+1/(n+1)` correction, so it prints `0.0000`.
  A permutation p-value can never legitimately be zero.
- `np.random` is never seeded: the suite is not reproducible run to run.
- Running it **overwrites the committed `reports/statistical_verification.png`** as a
  side effect.

### Other statistical issues

**Mann-Kendall on an autocorrelated level series** (`weekly_report.py:503`). The basis
`rack − nymex` has lag-1 autocorrelation of **0.845 (RB)** / **0.715 (HO)**. MK assumes
independence. Simulated rejection rate under a no-trend AR(1) null, nominal 5%:

| ρ | 0.0 | 0.5 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| actual rejection rate | 3.8% | 25.0% | 47.2% | 61.0% | 72.0% |

At the observed ρ the "significant basis drift" warning is a **~50% false-alarm rate**.
Current p-values are 0.084 / 0.097 — just under the wire, so the warning will flicker on
and off arbitrarily. Use a pre-whitened (Hamed-Rao) variance or test the differenced
series. Minor: the window is labelled "90-day" but uses 65 rows.

**CVaR from 5 observations** (`backtest.py:373`). The comment claims 40 WAIT-days gives
"at least 2 data points" for stability. Reproduced:

```
RB: 93 wait-days, tail n=5, values [1.27 1.44 1.75 2.53 6.53] -> CVaR 2.70¢
HO: 92 wait-days, tail n=5, values [0.63 1.65 1.76 2.22 8.49] -> CVaR 2.95¢
```

Dropping the single worst RB point moves the estimate from 2.70¢ to 1.75¢ — a 35% swing
from one observation. This number is quoted to the user in dollars per truck.

**Conviction ladder is not supported for RB.** From `metrics_cache.json`:

| | n | win rate | 95% CI |
|---|---|---|---|
| RB low | 283 | 76.3% | [70.9, 81.2] |
| RB mod | 62 | **85.5%** | [74.2, 93.1] |
| RB high | 58 | 82.8% | [70.6, 91.4] |

RB high-vs-low **p = 0.371**; high-vs-mod **p = 0.874**. The ladder is non-monotone —
"Moderate" outscores "High". Yet `main.py:762` tells the user *"Low Conviction — do not
act on this signal unless inventory forces you to order regardless"*, suppressing action
on 283 of 403 alerts based on a difference that is not there. For **HO the ladder is
genuine** (high-vs-low p < 0.001, high-vs-mod p = 0.013) — apply the gate there, not to RB.

Also: `z = change / std` never subtracts the mean. Harmless here (daily mean ≈ 0) but it
is not a Z-score.

**Grid search.** The objective is essentially flat — all 64 combinations land within
**2.7%** of the best (430.8 → 442.3):

```
best=442.3  median=438.1  worst=430.8   |  combos within 5% of best: 64/64
```

So the "optimization" is selecting noise; harmless, but it is not the overfitting defence
the README claims, because the hyperparameters are still chosen by maximizing the same
three folds later reported as out-of-sample. The tie-break only compares `W`, so it can
adopt a worse `Hp`/`Dp` from a tying candidate.

**`BLEND_ALPHA = 0.3`** gives the threshold EWMA a half-life of **1.9 sessions** (95% of
weight inside 9 sessions). It provides almost none of the stability the README implies;
what actually stabilizes the thresholds is the 240-day percentile window underneath.

---

## 5. README claims vs. what the data says

### 5a. The performance table is not reproducible

Re-running the model's own logic with the live thresholds:

| | README | reproduced |
|---|---|---|
| RB 2023 | 34 alerts, 52.94%, +29.96¢ | **131 alerts, 62.60%, +233.01¢** |
| RB 2024 | 146 alerts, 54.11%, +61.31¢ | **140 alerts, 55.71%, +75.40¢** |
| RB 2025 | 158 alerts, 72.78%, +217.06¢ | 148 alerts, 76.35%, +216.78¢ |

No script in the repository regenerates this table.

### 5b. Internal contradictions

- HO: *"Honest Multi-Year Precision Envelope: 60%–79% (with an overall historical
  average of 94.8%)"* — an average **outside its own stated envelope**, and impossible
  given the yearly figures quoted two lines later (59.52%, 78.65%).
- RB: *"average savings of +6.04¢/gal per active alert"* — the same section's yearly
  table implies (29.96+61.31+217.06)/338 = **0.91¢**. A 7× discrepancy.
- RB: *"overall historical average of 71.0%"* — the yearly table weights to **62.7%**.
- README: grid is `W∈{120,180,240}, Hp∈{15,20}, Dp∈{80,85}` = 12 combos. Code default is
  `{90,120,180,240} × {10,15,20,25} × {75,80,85,90}` = **64 combos**.
- README: *"over the last 365 days of history"*. With W=240 the earliest fold's training
  window starts 511 rows back — roughly **three years**.
- README: *"54 tests"* in `comprehensive_test_suite.py`. The file is `test_comprehensive.py`
  and the suite has **176 tests**.
- README: ingestion *"8:00 PM to 12:00 AM CT"*; the workflow comment says 7 PM; the
  actual cron (`0 0,1,2,3,4,5,13,14`) is 7 PM–midnight CDT and **6–11 PM CST**, losing
  the midnight retry for five months a year. The `13,14` UTC entries are Tue–Sat CT
  mornings, not "Mon-Fri" as commented.

### 5c. The volatility narrative is backwards

> **CAUTION:** *"If the 2025 precision increase is primarily a result of calm markets…
> a return of high-volatility spikes in 2026 could cause performance to revert to the
> conservative planning floor of 53% / 60%."*

Mean absolute daily move by year:

| year | RB rack | HO rack | RB NYMEX | HO NYMEX | RB precision | HO precision |
|---|---|---|---|---|---|---|
| 2023 | 4.13¢ | 5.51¢ | 4.80¢ | 5.29¢ | 62.6% | 54.6% |
| 2024 | 2.79¢ | 3.63¢ | 3.64¢ | 3.66¢ | 55.7% | 61.5% |
| 2025 | 2.33¢ | 3.73¢ | 3.13¢ | 3.99¢ | 76.4% | 77.4% |
| 2026 | **5.70¢** | **11.02¢** | **7.51¢** | **11.55¢** | **96.6%** | **94.6%** |

**2025 was the calm year. 2026 is the violent one** (HO daily moves 3× 2024, singles up to
67¢), and precision went **up**, not down. High volatility raises precision here because
the pass-through signal grows relative to the fixed rack noise floor. The stated causal
story — and the risk framing built on it — is inverted.

### 5d. 2026 performance is near-oracle, which is the tell

| | model "savings" | Σ abs daily rack move | % of total movement captured |
|---|---|---|---|
| RB 2026 | +910¢ | 964¢ | **94%** |
| HO 2026 | +1796¢ | 1862¢ | **96%** |

Perfect foresight captures 100%. A model capturing 96% of every day's absolute move is
not forecasting — it is reading a rack that is a near-deterministic same-day function of
the settle (HO lag-0 correlation 0.94). The economic value therefore rests entirely on a
**contractual** question, not a statistical one: will Graves actually fill an order at the
prior day's price after 2:35 PM? If not, the edge is zero regardless of the R².

### 5e. "Rockets and Feathers" holds for RB only — not HO

Fitting `rackΔ = a + b_up·max(nymexΔ,0) + b_dn·min(nymexΔ,0)`:

| series | era | b_up | b_dn | b_up − b_dn | p | verdict |
|---|---|---|---|---|---|---|
| RB | 2025-10+ | 0.734 | 0.530 | +0.205 | **0.005** | supported |
| HO | 2025-10+ | 0.900 | 0.937 | −0.036 | 0.527 | **not supported** (sign reversed) |
| RB | full | 0.471 | 0.413 | +0.058 | 0.357 | not supported |
| HO | full | 0.775 | 0.768 | +0.007 | 0.912 | not supported |

The README presents asymmetric pass-through as the durability mechanism for **both**
commodities and quotes an HO ratio of 1.06×–1.50×. The observed HO win/loss ratio is a
consequence of the asymmetric *thresholds* (+1.62¢ vs −1.56¢) in a trending market, not
of asymmetric pass-through.

### 5f. The dollar figures assume implausible volume

`savings_cents.sum() / 100 × 8500` treats every alert as a full 8,500-gallon truck:

```
RB 2026: 146 alerts -> 1,241,000 gal
HO 2026: 149 alerts -> 1,266,500 gal
combined implied throughput: 2,507,500 gal/yr
```

A typical single-site independent station moves ~1.0–1.5M gal/yr **total**. The implied
volume is roughly 2× that, split across two products. The README's headline
*"$46,072.55 OOS savings"* also omits the `DISPATCH_SAME_DAY_RATE = 0.50` haircut that
`weekly_report.py` itself applies.

Deeper: summing consecutive daily deltas assumes you can buy ahead *every* day, which
requires unbounded storage. Real savings are bounded by tank turns, not alert count.

---

## 6. Integrity and engineering

**Tamper detection on the track-record file is disabled.** `validate_data.py:596` — when
`prediction_log.csv`'s historical prefix hash mismatches, the code does not fail; it
re-records a fresh hash with the comment *"a same-count hash mismatch here is a format
normalization, not tampering."* Any edit to any historical row is silently accepted. The
README says the registry enforces "strict immutability" and protects against "manual
edits"; for the file holding the performance record, it does not. (It *is* enforced for
`graves_history.csv`.) Also, the records carry no `prev_hash` link — "blockchain-style"
overstates a list of independent snapshots, and `integrity_hashes.csv` is itself
unanchored.

By contrast, **`calibration_artifacts.py` is genuinely good** — a real hash-chained,
append-only ledger with `prior_artifact_id` linkage and chronology enforcement. It is
the strongest component in the repo.

**But the chain that is verified is not the chain that is used.** `backtest.py:main()`
writes live thresholds to the mutable `metrics_cache.json` *and separately* builds a
shadow artifact from a different seed config (`calibration_seed_cfg`) anchored on the
prior *artifact*. Two parallel EWMA chains that can drift. `replay_day.py` replays
against the artifact — i.e. it validates a calibration the live system did not use.

**`replay_day.py` currently exits 1 on a normal day:**

```
$ python3 replay_day.py --date 2026-09-11
[RB] => SUCCESS: Replay matches logged prediction perfectly.
[RB] WARNING: NYMEX change cents difference of 0.4400 detected!
AUDIT FAILED: 2 mismatch/leakage warnings detected.
```

The 0.44¢/0.56¢ gap is the §3 snapshot-vs-settle difference — expected, not a leak. The
script counts it as a leakage failure. It is not wired into CI, so nobody sees it. Its
closing line, *"100% deterministic and leakage-free"*, is not supported by anything it
tests: it only checks that a threshold comparison reproduces a logged direction.

**`pandas` 3 breaks the outcome backfill.** `weekly_report.py:273` assigns a float into a
string-dtype column:

```
TypeError: Invalid value '5.0' for dtype 'str'
FAILED test_comprehensive.py::TestCategory15PredictionLog::test_15_2_pending_backfill_correct_direction
1 failed, 175 passed
```

CI pins `pandas==2.2.1` so it is green there, but this is the code path that resolves
every PENDING outcome.

**Dead in-sample metrics.** `RB_historical_win_rate` (0.9436) and `RB_average_savings`
(5.1889) are computed in-sample on the final training window, written to
`metrics_cache.json`, and **never read by anything**. A reader would reasonably take
them for the system's performance.

**Smaller items**
- `validate_data.py:208` — the >$1.00 daily-jump check runs on `.dropna()`'d rows, so a
  diff can span a multi-day gap yet be tested against a one-day limit. The comment
  justifying the threshold ("clamped to 3.0 cents… absolute statistical robustness")
  only covers threshold fitting; a bad print still flows straight into savings, win rate,
  CVaR and `nymex_daily_std`.
- `PRICE_MAX` is 6.00 in ingest/config but 10.00 in `validate_graves_history`. Observed
  `rack_d` already reached 5.47 — ~10% headroom before ingestion starts rejecting valid
  prices.
- `main.py:762` hardcodes `"53%–73%"` / `"60%–79%"` into live alert text, and
  `weekly_report.py:605` draws 53%/60% floor lines — all sourced from the unreproducible
  README table.
- Colour semantics are inverted between views: `build_rack_signal` uses red for "Hike
  likely", while `get_morning_confirmation_html:1182` colours a rack *increase* green.
- Roll-day suppression is disabled under pytest (`main.py:660`), so tests never exercise
  the production path — which is part of why §2 went unnoticed.
- Live coverage gap: 84 live rows over ~80 business days = **~42 sessions**, roughly half.
  If workflow failures correlate with unusual market conditions the live record is
  selectively sampled.
- `pyflakes`: `_prev_session` (main.py:677), `momentum_html` (1039), `header` (1981),
  `unlabelled_df`/`live_summary` (weekly_report.py:315-316) assigned but unused; ~13
  unused imports in `main.py`.
- `tracker.yml` runs every 5 minutes 24/7 = ~8,640 runs/month.
- `weekly_report.mann_kendall_test` returns tau-a (no tie correction) while computing a
  tie-corrected variance — inconsistent.

---

## What is solid

- `calibration_artifacts.py` — real hash-chained point-in-time ledger.
- `date_block_permutation_test` — correctly preserves same-day RB/HO dependence by
  permuting whole session blocks within coverage strata. Properly uses `(k+1)/(n+1)`.
- `weekly_report.py` restricts all operational metrics and the significance test to
  `prediction_source == 'live'` and excludes backfilled rows. Good discipline — the
  README does not honour it.
- Backfill arithmetic is exact: 1,368 resolved outcomes recomputed from history,
  **0 mismatches**.
- The 3:2:1 crack spread algebra is correct: `(2·RB·42 + 1·HO·42 − 3·CL)/3 = 28·RB + 14·HO − CL`.
- `crude_last_trade_date` correctly implements the CME rule (3 business days before the
  25th, 4 if the 25th is not a business day).
- 176 tests, and the purged walk-forward fold construction contains no train/test overlap.

---

## Priority

1. Fix the date convention in `graves_history.csv` (§1), then recompute everything.
2. Fix `is_contract_roll_day` (§2) — change `is_contract_roll_day` to resolve the LTD
   with `early_roll_days=0`, and move `scratch/verify_roll_days.py` into the CI suite.
3. Reconcile the live snapshot against the official settle (§3), or widen thresholds to
   exceed the snapshot error, or suppress alerts within one snapshot-error of a boundary.
4. Delete or repair the four tautological checks in `verify_statistics.py` (§4); seed the
   RNG; stop overwriting the committed PNG.
5. Rewrite the README performance section from a script that regenerates it, restricted
   to the correctly aligned window, with CIs, and with the volume assumption stated at
   every dollar figure.
6. Restore real tamper detection on `prediction_log.csv` (§6).
