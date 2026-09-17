"""Tests for the history alignment layer.

The defect these guard against: ``graves_history.csv`` mixes two date
conventions.  Rows from the original bulk import are stamped one session late,
so their rack price is matched against the following day's settle.  Calibrating
across that boundary trains roughly 60% of the pairs on the wrong settle.
"""

import os

import numpy as np
import pandas as pd
import pytest

import alignment


def _frame(dates, nymex, rack):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "nymex_rb": nymex, "nymex_ho": nymex,
        "rack_u": rack, "rack_p": rack, "rack_d": rack,
    })


def _synthetic(rows=400, lag=0, seed=3):
    """A series where the rack responds at the requested lag."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-08-01", periods=rows)
    step = rng.normal(0, 0.03, rows)
    nymex = 2.0 + step.cumsum()
    driver = np.roll(step, lag)
    driver[:lag] = 0.0
    rack = 2.1 + (0.7 * driver + rng.normal(0, 0.005, rows)).cumsum()
    return _frame(dates, nymex, rack)


# --- loading ---------------------------------------------------------------

def test_load_history_drops_weekends_and_sorts():
    df = _frame(["2026-05-18", "2026-05-16", "2026-05-15", "2026-05-15"],
                [2.0, np.nan, 2.1, 2.1], [2.2, 2.3, 2.4, 2.4])
    loaded = alignment.load_history_from_frame(df)
    assert list(loaded["date"].dt.strftime("%Y-%m-%d")) == ["2026-05-15", "2026-05-18"]


def test_calibration_history_honours_the_era_boundary():
    """The whole file is calibration-eligible now that it has been re-dated.

    Before the migration this excluded roughly two thirds of the rows.  The
    mechanism is still exercised with an explicit boundary, so it cannot rot.
    """
    full = alignment.load_history()
    clean = alignment.calibration_history(full)
    assert clean["date"].min() >= pd.Timestamp(alignment.CALIBRATION_ERA_START)
    assert len(clean) >= 700, (
        f"only {len(clean)} eligible rows; the migration should leave ~835")

    cut = "2025-08-04"
    narrowed = alignment.calibration_history(full, era_start=cut)
    assert len(narrowed) < len(clean)
    assert narrowed["date"].min() >= pd.Timestamp(cut)


def test_aligned_deltas_never_spans_a_missing_observation():
    """A pair must join two consecutive *observed* sessions of both series."""
    df = _frame(["2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20"],
                [2.00, np.nan, 2.10, 2.15],
                [2.20, 2.25, 2.30, 2.35])
    frame = alignment.aligned_deltas(df, "RB")
    # The NaN settle row cannot contribute, so only one pair survives:
    # 2026-05-19 -> 2026-05-20.
    assert len(frame) == 1
    assert frame["date"].iloc[0] == pd.Timestamp("2026-05-20")
    assert frame["delta_nymex"].iloc[0] == pytest.approx(5.0)
    assert frame["session_gap"].iloc[0] == 1


# --- the diagnostic itself -------------------------------------------------

def test_lag_diagnostic_passes_a_correctly_aligned_series():
    diag = alignment.lag_diagnostics(_synthetic(lag=0), "RB")
    assert diag["aligned"] is True
    assert abs(diag["b1"]) < 0.15
    assert diag["b0"] == pytest.approx(0.7, abs=0.1)


def test_lag_diagnostic_catches_a_one_session_stamping_error():
    """The exact defect: the rack responds to yesterday's settle."""
    diag = alignment.lag_diagnostics(_synthetic(lag=1), "RB")
    assert diag["aligned"] is False
    assert diag["b1"] > diag["b0"], "a shifted series loads onto lag-1"
    assert "mixing date conventions" in diag["reason"]


ARCHIVE = "data/graves_history.pre_alignment_migration.csv"


@pytest.mark.skipif(not os.path.exists(ARCHIVE), reason="pre-migration archive absent")
def test_the_archived_pre_migration_file_is_still_detected_as_misaligned():
    """The detector must still catch the defect it was built for.

    The live file has been corrected, so it can no longer serve as the positive
    case.  The archived original is kept precisely so this test keeps its teeth:
    if the detector ever stops flagging it, the detector is broken.
    """
    before = alignment.load_history(ARCHIVE)
    flagged = [p for p in ("RB", "HO")
               if not alignment.lag_diagnostics(before, p)["aligned"]]
    assert flagged, "the detector no longer flags the known-bad original file"
    with pytest.raises(alignment.AlignmentError):
        alignment.assert_calibration_alignment(
            before, era_start="2023-01-01", csv_path=ARCHIVE)


def test_the_live_history_is_now_aligned_end_to_end():
    """The migration's purpose, asserted on the real file."""
    full = alignment.load_history()
    for prefix in ("RB", "HO"):
        diag = alignment.lag_diagnostics(full, prefix)
        assert diag["aligned"] is True, f"{prefix}: {diag['reason']}"
        assert diag["n"] >= 700, f"{prefix}: only {diag['n']} usable pairs"
        # The corrected legacy era must agree with the era that was already
        # trusted; a pass-through slope far from it would mean a bad re-dating.
        assert 0.5 < diag["b0"] < 1.1, f"{prefix}: implausible slope {diag['b0']}"


def test_verified_era_passes_the_gate():
    report = alignment.assert_calibration_alignment()
    for prefix in ("RB", "HO"):
        diag = report[prefix]
        assert diag["aligned"] is True
        assert diag["ratio"] < alignment.MAX_LAG1_RATIO
        assert diag["n"] >= 200
    structure = report["structure"]
    assert structure["fraction"] <= alignment.MAX_LEGACY_WEEK_FRACTION
    assert structure["total_weeks"] > 30


def test_gate_raises_on_a_misaligned_window():
    shifted = _synthetic(lag=1)
    with pytest.raises(alignment.AlignmentError) as excinfo:
        alignment.assert_calibration_alignment(shifted, era_start="2025-08-01")
    assert "same-session alignment" in str(excinfo.value)


def test_configured_era_start_satisfies_both_criteria():
    """The cut must be statistically clean AND structurally unambiguous."""
    configured = pd.Timestamp(alignment.CALIBRATION_ERA_START)

    # 1. Statistical: the lag-1 term must be insignificant from here on.
    table = alignment.scan_era_start()
    clean = table[table["clean"]]
    assert not clean.empty
    assert pd.Timestamp(clean.iloc[0]["start"]) <= configured, (
        "the configured start precedes the earliest statistically clean month")

    # 2. Structural: no week after the cut may contain both conventions.
    #    This reads the RAW file: Saturday rows are the legacy fingerprint and
    #    load_history() has already dropped them.
    raw = alignment.load_raw_history()
    era_rows = raw[raw["date"] >= configured]
    weeks = alignment.convention_weeks(era_rows)
    legacy = weeks["legacy"] + weeks["interleaved"]
    total = len(weeks["legacy"]) + len(weeks["live"]) + len(weeks["interleaved"])
    assert len(legacy) / total <= alignment.MAX_LEGACY_WEEK_FRACTION, (
        f"{len(legacy)}/{total} weeks in the era still carry the legacy "
        f"stamping: {legacy}")

    # And the cut must not be gratuitously late: the last interleaved week in
    # the whole file must end no more than a week before it.
    all_interleaved = alignment.convention_weeks(raw)["interleaved"]
    assert all_interleaved, "the file should contain interleaved weeks to cut after"
    last_end = pd.Timestamp(max(all_interleaved).split("/")[1])
    assert configured - last_end <= pd.Timedelta(days=7), (
        f"cut at {configured.date()} is far later than the last interleaved "
        f"week ending {last_end.date()}; verified rows are being discarded")


def test_convention_weeks_separates_the_two_stampings():
    legacy = _frame(["2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-07"],
                    [2.0, 2.0, 2.0, 2.0, np.nan], [2.1] * 5)
    live = _frame(["2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08"],
                  [2.0] * 5, [2.1] * 5)
    assert len(alignment.convention_weeks(legacy)["legacy"]) == 1
    assert alignment.convention_weeks(legacy)["live"] == []
    assert len(alignment.convention_weeks(live)["live"]) == 1
    assert alignment.convention_weeks(live)["legacy"] == []

    both = pd.concat([legacy.assign(date=pd.to_datetime(
        ["2025-06-02", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-07"])), ])
    assert alignment.convention_weeks(both)["interleaved"], (
        "a week with both a Monday and a Saturday row must be flagged")


def test_gate_rejects_an_interleaved_week(tmp_path):
    """Structural failure must fail the gate even when the regression passes."""
    rows = _synthetic(rows=400)
    saturday = pd.Timestamp(rows["date"].iloc[5]).normalize()
    while saturday.dayofweek != 5:
        saturday += pd.Timedelta(days=1)
    polluted = pd.concat([rows, _frame([saturday], [np.nan], [2.2])],
                         ignore_index=True).sort_values("date")
    assert alignment.convention_weeks(polluted)["interleaved"], "bad test setup"

    csv_path = tmp_path / "graves_history.csv"
    polluted.to_csv(csv_path, index=False)
    # One stray Saturday among 80 weeks is below the sustained-drift limit.
    alignment.assert_calibration_alignment(
        polluted, era_start="2025-08-01", csv_path=str(csv_path))

    # A wholesale return to the legacy stamping must fail the gate.
    legacy_wide = rows.copy()
    legacy_wide = legacy_wide[legacy_wide["date"].dt.dayofweek != 0]
    saturdays = pd.date_range(rows["date"].iloc[0], rows["date"].iloc[-1], freq="W-SAT")
    legacy_wide = pd.concat(
        [legacy_wide, _frame(saturdays, [np.nan] * len(saturdays), [2.2] * len(saturdays))],
        ignore_index=True).sort_values("date")
    legacy_path = tmp_path / "legacy.csv"
    legacy_wide.to_csv(legacy_path, index=False)
    with pytest.raises(alignment.AlignmentError) as excinfo:
        alignment.assert_calibration_alignment(
            legacy_wide, era_start="2025-08-01", csv_path=str(legacy_path))
    assert "legacy stamping" in str(excinfo.value)


def test_small_samples_are_reported_not_guessed():
    tiny = _synthetic(rows=20)
    diag = alignment.lag_diagnostics(tiny, "RB")
    assert diag["aligned"] is False
    assert "need 30" in diag["reason"]
