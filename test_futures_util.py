"""Regression tests for contract-roll detection.

These exist because ``is_contract_roll_day`` silently returned False for every
date of every year after ``early_roll_days`` was introduced, which disabled the
roll exclusion in calibration, in out-of-sample scoring, in the tail-risk
estimate and in live alert suppression at the same time.  The only guard that
would have caught it lived in ``scratch/``, which ``pytest.ini`` excludes.
"""

from datetime import date, timedelta

import numpy as np
import pytest

import alignment
import futures_util as fu


def _roll_sessions(year, prefix):
    sessions, day = [], date(year, 1, 1)
    while day <= date(year, 12, 31):
        if fu.is_contract_roll_day(day, prefix):
            sessions.append(day)
        day += timedelta(days=1)
    return sessions


@pytest.mark.parametrize("year", [2024, 2025, 2026])
@pytest.mark.parametrize("prefix", ["RB", "HO", "CL"])
def test_roll_day_fires_exactly_once_per_contract_month(year, prefix):
    """The core regression: it must fire, and fire the right number of times."""
    sessions = _roll_sessions(year, prefix)
    assert len(sessions) == 12, (
        f"{prefix} {year}: expected one roll session per contract month, "
        f"got {len(sessions)}")
    assert len({(d.year, d.month) for d in sessions}) == 12


@pytest.mark.parametrize("prefix", ["RB", "HO", "CL"])
def test_roll_day_never_fires_on_a_non_session(prefix):
    """A weekend or holiday is not a session and cannot be a roll session."""
    for day in _roll_sessions(2026, prefix):
        assert fu.is_nymex_business_day(day), f"{prefix}: {day} is not a session"


def test_roll_day_marks_the_session_where_the_contract_changes():
    """The flagged session is precisely the one whose diff spans two contracts."""
    day = date(2026, 1, 1)
    while day <= date(2026, 12, 31):
        if not fu.is_nymex_business_day(day):
            day += timedelta(days=1)
            continue
        previous = fu.previous_nymex_business_day(day)
        changed = (fu.get_front_month_contract(day, "RB")[:2]
                   != fu.get_front_month_contract(previous, "RB")[:2])
        assert fu.is_contract_roll_day(day, "RB") == changed, day
        day += timedelta(days=1)


def test_session_after_a_roll_is_clean():
    for roll in _roll_sessions(2026, "RB"):
        following = roll + timedelta(days=1)
        while not fu.is_nymex_business_day(following):
            following += timedelta(days=1)
        assert not fu.is_contract_roll_day(following, "RB")


def test_refined_product_ltd_is_the_last_business_day_of_the_prior_month():
    assert fu.contract_last_trade_date(2026, 6, "RB") == date(2026, 5, 29)
    assert fu.contract_last_trade_date(2026, 1, "HO") == date(2025, 12, 31)


def test_crude_ltd_is_three_business_days_before_the_25th():
    # 2026-05-25 is a Monday and a Memorial Day holiday, so the rule counts
    # back four business days from the 25th rather than three.
    ltd = fu.contract_last_trade_date(2026, 6, "CL")
    assert ltd < date(2026, 5, 25)
    assert fu.is_nymex_business_day(ltd)


def test_roll_offset_isolates_the_observed_settle_gap():
    """``DEFAULT_EARLY_ROLL_DAYS`` is measured, not assumed.

    A correct offset flags the sessions where NYMEX gaps but the rack does not
    follow.  Scored by the mean absolute residual of the rack regression, the
    calibrated value must separate flagged from unflagged sessions; a wrong
    offset does not.
    """
    df = alignment.calibration_history()
    separations = {}
    for offset in (1, 2, 3):
        ratios = []
        for prefix in ("RB", "HO"):
            frame = alignment.aligned_deltas(df, prefix)
            slope, intercept = np.polyfit(frame["delta_nymex"], frame["delta_rack"], 1)
            resid = (frame["delta_rack"] - (intercept + slope * frame["delta_nymex"])).abs()
            flagged = frame["date"].apply(
                lambda d, o=offset, p=prefix: fu.is_contract_roll_day(d.date(), p, early_roll_days=o))
            if flagged.sum() == 0:
                ratios.append(0.0)
                continue
            ratios.append(resid[flagged].mean() / resid[~flagged].mean())
        separations[offset] = min(ratios)

    best = max(separations, key=separations.get)
    assert best == fu.DEFAULT_EARLY_ROLL_DAYS, (
        f"the configured early-roll offset ({fu.DEFAULT_EARLY_ROLL_DAYS}) no longer "
        f"isolates the settle gap best; measured separations {separations}")
    assert separations[best] > 2.0, (
        "flagged roll sessions should carry a markedly larger rack residual")
