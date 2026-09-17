"""Single source of truth for pairing NYMEX settles with Graves rack prices.

Why this module exists
----------------------
``data/graves_history.csv`` contains two incompatible date conventions.

Rows written by the live ingest (``ingest_prices.py``) are stamped with the
session date ``D``: ``nymex_*`` is the 1:30 PM CT settle of ``D`` and ``rack_*``
is the price Graves posted the evening of ``D``.  That is the convention the
model assumes (``LAG_DAYS = 0``).

Rows written by the original bulk import (``historical_import/
extract_emails_to_csv.py``) were stamped from the email ``Date:`` header without
converting it to America/Chicago.  An 8 PM CT email carries a UTC timestamp of
01:00 the following day, so those rows landed one calendar day late: their
``rack_*`` belongs to session ``D`` while their ``nymex_*`` is the settle of
``D + 1``.

The fingerprint is unmistakable.  Regressing the daily rack change on the
current and previous NYMEX change (``rackD_t = a + b0*nymexD_t + b1*nymexD_t-1``)
over the full file gives a significant ``b1``; the legacy rows respond to
*yesterday's* settle.  The day-of-week histogram shows the same thing: 2023 has
42 Saturday rows and zero Monday rows, because every row is shifted forward one
day.

Why the legacy rows are excluded rather than repaired
-----------------------------------------------------
Shifting a legacy row back one session requires the settle for that earlier
session.  Under the legacy stamping no row was ever stamped Monday, so Monday
settles were never backfilled and are simply absent from the file.  Only
Wednesday/Thursday/Friday sessions could be reconstructed, which would leave a
systematic weekday hole, and the 2024-12 to 2025-07 transition interleaves both
conventions in the same week, so individual rows there cannot be classified with
confidence.

Guessing wrong injects a one-session misalignment straight into the live
thresholds, so the legacy rows are retained in the file but excluded from
calibration.  ``CALIBRATION_ERA_START`` is the earliest date from which ``b1``
is not statistically distinguishable from zero for both commodities.

Recovering the legacy era is possible once Monday settles are backfilled from an
external source; see ``docs/history-alignment.md``.
"""

import os

import numpy as np
import pandas as pd

# Earliest session whose alignment is verified.  Two criteria, both required:
#
# 1. Statistical -- the lag-1 pass-through coefficient must be insignificant
#    (p > ERA_SELECTION_PVALUE) for both commodities.  On its own this would
#    permit a start as early as 2024-05.
# 2. Structural -- the cut must sit after the last week in which BOTH stamping
#    conventions appear.  A legacy week has a Saturday row and no Monday row; a
#    live week has a Monday row and no Saturday row.  Seven weeks contain both,
#    the last being 2025-07-28/2025-08-03, and individual rows inside them
#    cannot be classified with confidence.
#
# ``migrate_legacy_alignment.py`` re-dated the legacy rows on 2026-09-17, so
# both criteria are now met by the whole file and the boundary sits at its first
# session.  The gate below still runs on every calibration: it is what would
# catch the convention drifting again.  Re-derive with
# ``python3 -m alignment --scan``.
CALIBRATION_ERA_START = "2023-03-06"

# Runtime drift detector.  A correctly aligned series carries almost no lag-1
# pass-through; a series stamped one session late carries more lag-1 than lag-0.
# The test is deliberately on the RATIO b1/b0 rather than on b1 alone: as the
# clean era grows, a harmless residual b1 eventually becomes statistically
# significant, so significance on its own would turn into a permanent false
# alarm.  Materiality and significance must BOTH fire.
#
# Observed values for reference:
#   pre-migration file (conventions mixed)  RB b1/b0 = 0.31   HO b1/b0 = 0.11
#   after re-dating                         RB b1/b0 = 0.02   HO b1/b0 = 0.00
# 0.08 sits between them with room on both sides.  The pre-migration file is
# archived and test_alignment.py asserts it is still flagged.
MAX_LAG1_RATIO = 0.08
DRIFT_PVALUE = 0.01

# Criterion used to CHOOSE CALIBRATION_ERA_START: the earliest month from which
# the lag-1 term is not significant at 5% for either commodity.
ERA_SELECTION_PVALUE = 0.05

# Structural drift detector.  A legacy-stamped week has a Saturday row and no
# Monday row.  Isolated instances are expected and harmless -- a Friday email
# that arrived after midnight lands on Saturday, the Friday row goes missing,
# and both the weekend filter and the one-session contiguity filter discard the
# affected pairs.  A *sustained* rate means the convention itself has changed.
# The legacy era ran at ~100%; the verified era sits near 3%.
MAX_LEGACY_WEEK_FRACTION = 0.15

COMMODITY_COLUMNS = {
    "RB": ("nymex_rb", "rack_u"),
    "HO": ("nymex_ho", "rack_d"),
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")


class AlignmentError(RuntimeError):
    """Raised when the history cannot be trusted for calibration."""


def load_raw_history(csv_path=None):
    """Load the file as written, weekends included.

    ``convention_weeks`` must see Saturday rows -- they are the fingerprint of
    the legacy stamping.  Handing it weekend-filtered rows makes every week look
    live and turns the structural check into a silent no-op.
    """
    df = pd.read_csv(csv_path or CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_history(csv_path=None):
    """Load the full history as sorted, weekday-only, de-duplicated rows.

    Weekend rows are dropped here rather than in each caller.  Every legacy
    Saturday row carries a NaN settle, so it can never contribute a usable pair;
    dropping it up front keeps ``diff()`` from spanning it and silently turning a
    one-session rack move into a multi-session one.
    """
    df = pd.read_csv(csv_path or CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.dayofweek < 5]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def load_history_from_frame(df):
    """Normalise an already-loaded frame the same way ``load_history`` does.

    Callers that hold a raw frame (tests, the artifact builder) must go through
    the identical weekday/sort/dedupe path, otherwise a training set could be
    assembled with weekend rows that ``load_history`` would have removed.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out["date"].dt.dayofweek < 5]
    return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def calibration_history(df=None, era_start=CALIBRATION_ERA_START, csv_path=None):
    """Return only the rows whose NYMEX/rack alignment has been verified."""
    if df is None:
        df = load_history(csv_path)
    clean = df[df["date"] >= pd.Timestamp(era_start)].reset_index(drop=True)
    return clean


def aligned_deltas(df, prefix):
    """Return one-session (nymex_delta, rack_delta) pairs in cents.

    Rows are dropped, not forward-filled, when either price is missing, and the
    delta is then computed over the surviving rows.  That alone is not enough:
    if a session is absent -- a failed ingest, or one of the 121 rows with no
    settle -- the neighbours either side become adjacent and their difference
    silently becomes a *two*-session move.  The thresholds are one-session
    thresholds, so such a pair is not comparable and is discarded.

    A pair is kept only when the previous surviving row is the immediately
    preceding NYMEX business day.  Friday to Monday is one session and is kept;
    Friday to Tuesday with Monday missing is not.
    """
    nymex_col, rack_col = COMMODITY_COLUMNS[prefix]
    clean = df.dropna(subset=[nymex_col, rack_col]).sort_values("date")
    frame = pd.DataFrame({
        "date": clean["date"],
        "delta_nymex": clean[nymex_col].diff() * 100,
        "delta_rack": clean[rack_col].diff() * 100,
        "prior_date": clean["date"].shift(1),
    }).dropna(subset=["delta_nymex", "delta_rack", "prior_date"])

    if frame.empty:
        frame["session_gap"] = []
        return frame.reset_index(drop=True)

    from futures_util import previous_nymex_business_day
    expected = frame["date"].map(lambda d: previous_nymex_business_day(d.date()))
    contiguous = expected.to_numpy() == frame["prior_date"].dt.date.to_numpy()
    frame = frame[contiguous].copy()
    frame["session_gap"] = (frame["date"] - frame["prior_date"]).dt.days
    # prior_date is retained so consumers can tell whether two pairs are
    # themselves adjacent; lag_diagnostics needs that to build a valid lag-1
    # term after rows have been dropped.
    return frame.reset_index(drop=True)


def convention_weeks(df):
    """Classify each week by which stamping convention its rows show.

    The legacy import shifted every row one day forward, which removes Mondays
    and creates Saturdays; the live ingest does neither.  A week holding both a
    Saturday and a Monday row therefore contains rows from both conventions and
    cannot be safely used, regardless of what the regression says.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["week"] = work["date"].dt.to_period("W")
    weekdays = work.groupby("week")["date"].apply(lambda s: set(s.dt.dayofweek))
    legacy, live, interleaved = [], [], []
    for week, days in weekdays.items():
        has_saturday, has_monday = 5 in days, 0 in days
        if has_saturday and has_monday:
            interleaved.append(str(week))
        elif has_saturday:
            legacy.append(str(week))
        elif has_monday:
            live.append(str(week))
    return {"legacy": legacy, "live": live, "interleaved": interleaved}


def lag_diagnostics(df, prefix):
    """Fit ``rackD_t = a + b0*nymexD_t + b1*nymexD_{t-1}`` and report b1.

    ``b1`` is the alignment test.  A same-session series has b1 ~ 0; a series
    stamped one day late loads onto b1 instead of b0.
    """
    frame = aligned_deltas(df, prefix)
    x0 = frame["delta_nymex"]
    x1 = frame["delta_nymex"].shift(1)
    y = frame["delta_rack"]
    # The lag-1 term is only meaningful when the previous *pair* is the one
    # immediately before this one.  Dropped rows make shift(1) reach back to an
    # unrelated session, which destroys the diagnostic's power precisely in the
    # eras it needs to flag -- the legacy era loses one pair every week, so a
    # naive shift compares Wednesday against the previous Friday.
    adjacent = frame["prior_date"].eq(frame["date"].shift(1))
    mask = x0.notna() & x1.notna() & y.notna() & adjacent
    n = int(mask.sum())
    if n < 30:
        return {"n": n, "b0": float("nan"), "b1": float("nan"),
                "ratio": float("nan"), "p_b1": float("nan"), "aligned": False,
                "reason": f"only {n} usable pairs; need 30"}

    design = np.column_stack([np.ones(n), x0[mask].to_numpy(), x1[mask].to_numpy()])
    target = y[mask].to_numpy()
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    dof = n - design.shape[1]
    scale = float(resid @ resid) / dof
    cov = scale * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(cov[2, 2]))
    from scipy import stats  # local import keeps module import cheap for main.py
    p_b1 = float(2 * (1 - stats.t.cdf(abs(beta[2] / se), dof))) if se > 0 else 1.0

    ratio = abs(beta[2]) / abs(beta[1]) if beta[1] != 0 else float("inf")
    material = ratio > MAX_LAG1_RATIO
    significant = p_b1 < DRIFT_PVALUE
    aligned = not (material and significant)
    return {
        "n": n,
        "b0": float(beta[1]),
        "b1": float(beta[2]),
        "ratio": float(ratio),
        "p_b1": p_b1,
        "aligned": bool(aligned),
        "reason": "" if aligned else (
            f"lag-1 pass-through b1={beta[2]:+.3f} is {ratio:.0%} of the "
            f"same-session coefficient b0={beta[1]:+.3f} (p={p_b1:.4f}); the "
            "history is mixing date conventions again"
        ),
    }


def assert_calibration_alignment(df=None, era_start=CALIBRATION_ERA_START,
                                 csv_path=None):
    """Fail loudly if the calibration-eligible rows are not same-session aligned.

    This runs before every calibration.  It is the guard that would have caught
    the original defect: a one-session stamping error anywhere in the calibration
    window shows up as a significant lag-1 coefficient.
    """
    if df is None:
        df = load_history()
    clean = calibration_history(df, era_start)
    report = {}
    problems = []
    for prefix in COMMODITY_COLUMNS:
        diag = lag_diagnostics(clean, prefix)
        report[prefix] = diag
        if not diag["aligned"]:
            problems.append(f"{prefix}: {diag['reason']}")

    # Structural check, independent of the regression.  This must read the raw
    # file: Saturday rows are the legacy fingerprint and `clean` has already
    # dropped them.
    try:
        raw = load_raw_history(csv_path)
    except Exception:
        raw = None
    if raw is not None:
        era_rows = raw[raw["date"] >= pd.Timestamp(era_start)]
        weeks = convention_weeks(era_rows)
        legacy_weeks = weeks["legacy"] + weeks["interleaved"]
        total_weeks = len(weeks["legacy"]) + len(weeks["live"]) + len(weeks["interleaved"])
        fraction = len(legacy_weeks) / total_weeks if total_weeks else 0.0
        report["structure"] = {
            "legacy_weeks": len(legacy_weeks),
            "total_weeks": total_weeks,
            "fraction": fraction,
            "examples": legacy_weeks[:5],
        }
        if fraction > MAX_LEGACY_WEEK_FRACTION:
            problems.append(
                f"{len(legacy_weeks)} of {total_weeks} weeks "
                f"({fraction:.0%}) carry the legacy stamping (a Saturday row and "
                f"no Monday row), above the {MAX_LEGACY_WEEK_FRACTION:.0%} limit; "
                f"examples: {legacy_weeks[:5]}")

    if problems:
        raise AlignmentError(
            "Graves history failed the same-session alignment check. "
            + "; ".join(problems)
        )
    return report


def scan_era_start(df=None, candidates=None):
    """Report the lag-1 coefficient for a range of candidate era starts.

    ``clean`` marks the selection criterion used to pick
    ``CALIBRATION_ERA_START``: lag-1 insignificant at ``ERA_SELECTION_PVALUE``
    for both commodities.  That is deliberately stricter than the runtime drift
    detector, because here we are choosing where to cut rather than deciding
    whether an already-verified window has broken.
    """
    if df is None:
        df = load_history()
    if candidates is None:
        first = df["date"].min().to_period("M").to_timestamp()
        last = df["date"].max().to_period("M").to_timestamp()
        candidates = [d.date().isoformat()
                      for d in pd.date_range(first, last, freq="MS")]
    rows = []
    for start in candidates:
        sub = calibration_history(df, start)
        entry = {"start": start}
        ok = True
        for prefix in COMMODITY_COLUMNS:
            diag = lag_diagnostics(sub, prefix)
            entry[f"{prefix}_n"] = diag["n"]
            entry[f"{prefix}_b0"] = diag["b0"]
            entry[f"{prefix}_b1"] = diag["b1"]
            entry[f"{prefix}_p"] = diag["p_b1"]
            ok = ok and diag["n"] >= 30 and diag["p_b1"] >= ERA_SELECTION_PVALUE
        entry["clean"] = ok
        rows.append(entry)
    return pd.DataFrame(rows)


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="History alignment diagnostics")
    parser.add_argument("--scan", action="store_true",
                        help="Report the lag-1 coefficient by candidate era start")
    args = parser.parse_args()

    df = load_history()
    if args.scan:
        table = scan_era_start(df)
        pd.set_option("display.width", 160)
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        clean = table[table["clean"]]
        if not clean.empty:
            print(f"\nEarliest verified-clean era start: {clean.iloc[0]['start']}")
            print(f"Configured CALIBRATION_ERA_START:   {CALIBRATION_ERA_START}")
        return

    print(f"Full history:            {len(df)} weekday rows "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")
    clean = calibration_history(df)
    print(f"Calibration-eligible:    {len(clean)} rows from {CALIBRATION_ERA_START}")
    for prefix in COMMODITY_COLUMNS:
        full = lag_diagnostics(df, prefix)
        era = lag_diagnostics(clean, prefix)
        print(f"  {prefix} full history  n={full['n']:4d} b0={full['b0']:+.3f} "
              f"b1={full['b1']:+.3f} ratio={full['ratio']:.2f} "
              f"(p={full['p_b1']:.4f})  aligned={full['aligned']}")
        print(f"  {prefix} calibration   n={era['n']:4d} b0={era['b0']:+.3f} "
              f"b1={era['b1']:+.3f} ratio={era['ratio']:.2f} "
              f"(p={era['p_b1']:.4f})  aligned={era['aligned']}")
    assert_calibration_alignment(df)
    print("\nAlignment check: PASSED")


if __name__ == "__main__":
    _main()
