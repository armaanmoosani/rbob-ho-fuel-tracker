"""Re-date the legacy-stamped rows of graves_history.csv to their true session.

Background
----------
Rows written by the original bulk email import were stamped from the email
``Date:`` header without converting to America/Chicago, so an 8 PM CT email
landed on the following calendar day.  Those rows hold a rack price belonging
to session ``S`` alongside the settle of ``S + 1``.  See
``docs/history-alignment.md``.

What this does
--------------
For every row in a *legacy-stamped week*, it moves the row back to
``S = previous_nymex_business_day(stamp)`` and replaces its settle with the
settle of ``S``.

The replacement settle is taken from the file itself wherever possible: the row
already stamped ``S`` holds exactly that value, so the correction is a
byte-for-byte move of an existing string, not a re-derivation.  Only Mondays
need an external value, because the legacy stamping never produced a Monday row
and those settles were consequently never backfilled.

Safety
------
Nothing is written unless every gate passes.  The gates are deliberately
redundant, because this rewrites the system of record:

  A  the external source must reproduce every legacy settle in the file exactly
  B  the internal shift and the external value must agree wherever both exist
  C  no two rows may land on the same date
  D  rack values must be conserved exactly, as a multiset
  E  no row may be created or destroyed
  F  rows at or after CALIBRATION_ERA_START must be byte-identical
  G  the correction must actually fix the alignment (lag-1 -> 0)
  H  the rewritten file must pass validate_data.validate_graves_history

Usage
-----
    python3 migrate_legacy_alignment.py             # dry run: report only
    python3 migrate_legacy_alignment.py --apply     # write, after all gates pass
"""

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

import alignment
from futures_util import is_nymex_business_day, previous_nymex_business_day

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "graves_history.csv")
ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "data",
                            "graves_history.pre_alignment_migration.csv")
RACK_COLUMNS = ("rack_u", "rack_p", "rack_d")
NYMEX_COLUMNS = ("nymex_rb", "nymex_ho")
SYMBOLS = {"nymex_rb": "RB=F", "nymex_ho": "HO=F"}

# Tolerance for "the external source reproduces the file".  The legacy era
# matches to 0.000000 in practice; this is a guard, not a fudge factor.
SETTLE_TOLERANCE = 1e-6


class MigrationAborted(RuntimeError):
    """Raised when a safety gate fails.  Nothing has been written."""


def read_rows(path=CSV_PATH):
    """Read the CSV as raw strings so untouched fields stay byte-identical."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, [dict(row) for row in reader]


def classify_weeks(rows):
    """Split weeks by which stamping convention their day-of-week pattern shows.

    The legacy import shifted every row forward one calendar day, which removes
    Mondays and creates Saturdays.  A week holding both is a changeover week and
    cannot be classified row by row.

    A week holding neither is "ambiguous" on its own evidence, but a week that
    predates the very first live-stamped week must be legacy: the live
    convention did not exist yet.  Those are promoted, which is worth 27 rows.
    Ambiguous weeks inside the changeover period stay untouched.
    """
    by_week = {}
    for row in rows:
        stamp = date.fromisoformat(row["date"])
        by_week.setdefault(stamp.isocalendar()[:2], []).append(stamp)

    kinds = {}
    for week, days in by_week.items():
        weekdays = {d.weekday() for d in days}
        has_saturday, has_monday = 5 in weekdays, 0 in weekdays
        if has_saturday and has_monday:
            kinds[week] = "interleaved"
        elif has_saturday:
            kinds[week] = "legacy"
        elif has_monday:
            kinds[week] = "live"
        else:
            kinds[week] = "ambiguous"

    live_weeks = [w for w, k in kinds.items() if k == "live"]
    if live_weeks:
        first_live = min(live_weeks)
        for week, kind in list(kinds.items()):
            if kind == "ambiguous" and week < first_live:
                kinds[week] = "legacy"
    return kinds


def fetch_external_settles(first, last):
    """Daily settles from the same series that originally backfilled the file."""
    import yfinance as yf
    out = {}
    start = (first - timedelta(days=10)).isoformat()
    end = (last + timedelta(days=5)).isoformat()
    for column, symbol in SYMBOLS.items():
        history = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        if history.empty:
            raise MigrationAborted(f"no external settles returned for {symbol}")
        out[column] = {ts.date(): float(v) for ts, v in
                       zip(pd.to_datetime(history.index), history["Close"].values)}
        print(f"  {symbol}: {len(out[column])} settles "
              f"{min(out[column])} -> {max(out[column])}")
    return out


def plan_migration(rows, external, era_start):
    """Decide each row's fate.  Returns (plan, unresolved).

    ``unresolved`` lists legacy rows whose corrected settle cannot be sourced;
    a non-empty list aborts the migration rather than inventing a value.
    """
    kinds = classify_weeks(rows)
    by_stamp = {date.fromisoformat(r["date"]): r for r in rows}

    plan, unresolved = [], []
    for row in rows:
        stamp = date.fromisoformat(row["date"])
        week_kind = kinds[stamp.isocalendar()[:2]]

        if stamp >= era_start or week_kind != "legacy":
            plan.append({"row": row, "stamp": stamp, "session": stamp,
                         "action": "keep", "reason": (
                             "at or after the verified era" if stamp >= era_start
                             else f"{week_kind} week")})
            continue

        # The legacy stamp is the session plus exactly one CALENDAR day: an
        # 8 PM CT email carries a UTC timestamp of 01:00 the next date.  Using
        # the previous *business* day instead collides whenever a holiday
        # intervenes -- Good Friday 2023-04-07 sent both the Friday and the
        # Saturday row to Thursday 04-06.
        session = stamp - timedelta(days=1)

        if session.weekday() >= 5:
            # Only the three Sunday-stamped rows reach here, and each sits in a
            # week with no Saturday row, so their true session cannot be
            # established.  They are left where they are rather than guessed at;
            # the weekday filter drops them either way.
            plan.append({"row": row, "stamp": stamp, "session": stamp,
                         "action": "keep",
                         "reason": "corrected session would fall on a weekend"})
            continue

        settles, sources = {}, {}
        for column in NYMEX_COLUMNS:
            donor = by_stamp.get(session)
            if donor is not None and donor[column] not in ("", None):
                # The row already stamped `session` holds that session's settle.
                # Reuse its exact string: no reformatting, no re-derivation.
                settles[column] = donor[column]
                sources[column] = "internal"
            elif session in external[column]:
                settles[column] = f"{external[column][session]:.10f}".rstrip("0").rstrip(".")
                sources[column] = "external"
            elif not is_nymex_business_day(session):
                # A market holiday genuinely has no settle.  Empty is how the
                # file already represents that, and the pairing layer drops it.
                settles[column] = ""
                sources[column] = "holiday_no_settle"
            else:
                settles[column] = None
                sources[column] = "missing"

        if any(v is None for v in settles.values()):
            unresolved.append((stamp, session))
            plan.append({"row": row, "stamp": stamp, "session": session,
                         "action": "unresolved", "reason": "no settle available"})
            continue

        plan.append({"row": row, "stamp": stamp, "session": session,
                     "action": "redate", "settles": settles, "sources": sources,
                     "reason": "legacy week"})
    return plan, unresolved


def build_rows(plan, fieldnames):
    out = []
    for item in plan:
        row = dict(item["row"])
        if item["action"] == "redate":
            row["date"] = item["session"].isoformat()
            for column, value in item["settles"].items():
                row[column] = value
        out.append(row)
    out.sort(key=lambda r: r["date"])
    return out


# --- gates -----------------------------------------------------------------

def gate_a_external_reproduces_file(rows, external, era_start):
    worst = 0.0
    checked = 0
    for row in rows:
        stamp = date.fromisoformat(row["date"])
        if stamp >= era_start:
            continue
        for column in NYMEX_COLUMNS:
            if not row[column]:
                continue
            if stamp not in external[column]:
                raise MigrationAborted(
                    f"gate A: external source has no settle for {stamp} ({column})")
            deviation = abs(float(row[column]) - external[column][stamp])
            worst = max(worst, deviation)
            checked += 1
    if worst > SETTLE_TOLERANCE:
        raise MigrationAborted(
            f"gate A: external source disagrees with the file by up to {worst:.6f} "
            f"in the legacy era; it cannot be trusted to supply Monday settles")
    return f"{checked} legacy settles reproduced externally, max deviation {worst:.2e}"


def gate_b_internal_and_external_agree(plan, external):
    worst, checked = 0.0, 0
    for item in plan:
        if item["action"] != "redate":
            continue
        for column in NYMEX_COLUMNS:
            if item["sources"][column] != "internal" or not item["settles"][column]:
                continue
            if item["session"] not in external[column]:
                continue
            deviation = abs(float(item["settles"][column]) - external[column][item["session"]])
            worst = max(worst, deviation)
            checked += 1
    if worst > SETTLE_TOLERANCE:
        raise MigrationAborted(
            f"gate B: the settle taken from inside the file disagrees with the "
            f"external source by up to {worst:.6f}; the shift model is wrong")
    return f"{checked} internally-sourced settles cross-checked, max deviation {worst:.2e}"


def gate_c_no_collisions(new_rows):
    counts = Counter(r["date"] for r in new_rows)
    clashes = {d: n for d, n in counts.items() if n > 1}
    if clashes:
        raise MigrationAborted(f"gate C: {len(clashes)} date collisions: "
                               f"{sorted(clashes)[:5]}")
    return f"{len(new_rows)} rows land on {len(counts)} distinct dates"


def gate_d_rack_values_conserved(old_rows, new_rows):
    before = Counter(tuple(r[c] for c in RACK_COLUMNS) for r in old_rows)
    after = Counter(tuple(r[c] for c in RACK_COLUMNS) for r in new_rows)
    if before != after:
        lost = before - after
        gained = after - before
        raise MigrationAborted(
            f"gate D: rack values changed. {sum(lost.values())} lost, "
            f"{sum(gained.values())} invented. examples lost: {list(lost)[:3]}")
    return f"all {sum(before.values())} rack triples preserved exactly"


def gate_e_row_count(old_rows, new_rows):
    if len(old_rows) != len(new_rows):
        raise MigrationAborted(
            f"gate E: row count changed {len(old_rows)} -> {len(new_rows)}")
    return f"row count unchanged at {len(old_rows)}"


def gate_f_live_era_untouched(old_rows, new_rows, era_start):
    def snapshot(rows):
        return {r["date"]: tuple(sorted(r.items()))
                for r in rows if date.fromisoformat(r["date"]) >= era_start}
    before, after = snapshot(old_rows), snapshot(new_rows)
    if before != after:
        changed = [d for d in before if before.get(d) != after.get(d)]
        missing = [d for d in before if d not in after]
        raise MigrationAborted(
            f"gate F: the verified era was modified. changed={changed[:5]} "
            f"missing={missing[:5]}")
    return f"all {len(before)} rows at or after {era_start} are byte-identical"


def gate_g_alignment_is_fixed(new_rows, fieldnames, era_start):
    frame = pd.DataFrame(new_rows)[list(fieldnames)]
    for column in NYMEX_COLUMNS + RACK_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    legacy = alignment.load_history_from_frame(frame[frame["date"] < pd.Timestamp(era_start)])

    messages = []
    for prefix in ("RB", "HO"):
        diagnostic = alignment.lag_diagnostics(legacy, prefix)
        if diagnostic["n"] < 100:
            raise MigrationAborted(
                f"gate G: only {diagnostic['n']} corrected pairs for {prefix}")
        if not diagnostic["aligned"]:
            raise MigrationAborted(
                f"gate G: the corrected legacy era is STILL misaligned for "
                f"{prefix}: {diagnostic['reason']}. The re-dating premise is wrong; "
                f"nothing has been written.")
        messages.append(f"{prefix} b0={diagnostic['b0']:+.3f} b1={diagnostic['b1']:+.3f} "
                        f"(ratio {diagnostic['ratio']:.2f}, p={diagnostic['p_b1']:.3f}, "
                        f"n={diagnostic['n']})")
    return "corrected legacy era now aligned: " + "; ".join(messages)


def gate_h_validator_accepts(path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c",
         f"import validate_data,sys;validate_data.validate_graves_history({path!r})"],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise MigrationAborted(
            f"gate H: validate_graves_history rejected the rewritten file:\n"
            f"{result.stdout}{result.stderr}")
    return "validate_graves_history accepts the rewritten file"


# --- driver ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the corrected file (only if every gate passes)")
    parser.add_argument("--era-start", default=alignment.CALIBRATION_ERA_START)
    parser.add_argument("--rebaseline-only", action="store_true",
                        help="append the new graves_history.csv baseline to the "
                             "integrity registry and exit. Use after --apply.")
    parser.add_argument("--rebaseline-integrity", action="store_true",
                        help="also append a new baseline for graves_history.csv to "
                             "data/integrity_hashes.csv. Separate from --apply "
                             "because rewriting history and re-baselining the audit "
                             "registry are distinct authorisations.")
    args = parser.parse_args()
    era_start = date.fromisoformat(args.era_start)

    if args.rebaseline_only:
        if not os.path.exists(ARCHIVE_PATH):
            raise MigrationAborted(
                "refusing to re-baseline: no pre-migration archive exists, so "
                "there is nothing to show the change was an authorised migration")
        rebaseline_integrity_hash()
        return

    fieldnames, rows = read_rows()
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    kinds = classify_weeks(rows)
    tally = Counter(kinds.values())
    print(f"Week classification: {dict(tally)}")

    print("\nFetching external settles...")
    stamps = [date.fromisoformat(r["date"]) for r in rows]
    external = fetch_external_settles(min(stamps), max(stamps))

    plan, unresolved = plan_migration(rows, external, era_start)
    actions = Counter(item["action"] for item in plan)
    print(f"\nPlan: {dict(actions)}")
    if unresolved:
        raise MigrationAborted(
            f"{len(unresolved)} legacy rows have no available settle for their "
            f"corrected session, e.g. {unresolved[:5]}. Refusing to invent values.")

    sources = Counter(s for item in plan if item["action"] == "redate"
                      for s in item["sources"].values())
    print(f"Settle provenance for re-dated rows: {dict(sources)}")

    new_rows = build_rows(plan, fieldnames)

    print("\nSafety gates:")
    checks = [
        ("A external reproduces file", lambda: gate_a_external_reproduces_file(rows, external, era_start)),
        ("B internal vs external", lambda: gate_b_internal_and_external_agree(plan, external)),
        ("C no date collisions", lambda: gate_c_no_collisions(new_rows)),
        ("D rack values conserved", lambda: gate_d_rack_values_conserved(rows, new_rows)),
        ("E row count", lambda: gate_e_row_count(rows, new_rows)),
        ("F verified era untouched", lambda: gate_f_live_era_untouched(rows, new_rows, era_start)),
        ("G alignment fixed", lambda: gate_g_alignment_is_fixed(new_rows, fieldnames, era_start)),
    ]
    for name, check in checks:
        print(f"  [PASS] {name}: {check()}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        _preview(plan)
        return

    tmp_path = CSV_PATH + ".migrated"
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"  [PASS] H validator: {gate_h_validator_accepts(tmp_path)}")

    shutil.copy2(CSV_PATH, ARCHIVE_PATH)
    os.replace(tmp_path, CSV_PATH)
    print(f"\nWritten. Original archived at {ARCHIVE_PATH}")
    print("Next: python3 alignment.py --scan, then move CALIBRATION_ERA_START back "
          "and regenerate the cache and README.")


def rebaseline_integrity_hash():
    """Append a new baseline for graves_history.csv to the integrity registry.

    validate_and_update_hashes compares the file against the most recent
    recorded hash and exits non-zero when historical rows differ.  That is the
    correct behaviour and exactly what this migration triggers; left unhandled
    it fails the tracker, the ingest, the weekly report and CI on every run.

    The registry stays append-only: the pre-migration hash remains in the file
    as the record of what the data was, and the evidence that this change was
    authorised sits beside it in git -- the archived pre-migration CSV, this
    script, and the commit that carries both.
    """
    import hashlib
    registry = os.path.join(os.path.dirname(CSV_PATH), "integrity_hashes.csv")
    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.read().splitlines()
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    previous = None
    with open(registry, "r") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4 and row[1] == "graves_history.csv":
                previous = row[3]
    if previous == digest:
        print("  Integrity registry already matches; nothing appended.")
        return
    with open(registry, "a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            [pd.Timestamp.now().isoformat(), "graves_history.csv", len(lines), digest])
    print(f"  Integrity registry re-baselined for graves_history.csv")
    print(f"    previous hash (retained): {previous}")
    print(f"    new hash:                 {digest}")
    print(f"    lines: {len(lines)}   archived original: {os.path.basename(ARCHIVE_PATH)}")


def _preview(plan):
    redated = [i for i in plan if i["action"] == "redate"]
    print(f"\nFirst 8 of {len(redated)} re-datings:")
    for item in redated[:8]:
        row = item["row"]
        print(f"  {item['stamp']} ({item['stamp'].strftime('%a')}) -> "
              f"{item['session']} ({item['session'].strftime('%a')})  "
              f"rack_u={row['rack_u']}  nymex_rb {row['nymex_rb'] or 'NaN':>20} -> "
              f"{item['settles']['nymex_rb']}  [{item['sources']['nymex_rb']}]")


if __name__ == "__main__":
    try:
        main()
    except MigrationAborted as exc:
        print(f"\nMIGRATION ABORTED — nothing was written.\n  {exc}")
        sys.exit(1)
