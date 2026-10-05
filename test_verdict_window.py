"""The daily verdict must survive a temporary loss of CI capacity.

On 2026-10-05 GitHub could not allocate a hosted runner across the whole
2:35-2:59 PM CT window: every tracker run between 14:20 and 14:45 CT was
cancelled or failed.  The verdict only fired while the clock read 14:3x-14:5x,
so with a five-minute poll that was five chances, and the day produced no
verdict at all.

Retrying later costs nothing in accuracy: build_rack_signal reads the 1:30 PM
settlement snapshot, which is final, so a verdict sent at 3:40 PM carries the
same numbers and is still actionable because the rack is not posted until about
6:00 PM.
"""

from datetime import datetime

import pytest
import pytz

import main

TZ = pytz.timezone("America/Chicago")


def _eligible(hour, minute):
    return main.VERDICT_WINDOW_START <= (hour, minute) < main.VERDICT_WINDOW_END


class TestWindowBoundaries:
    def test_opens_at_the_settlement_verdict_time(self):
        assert not _eligible(14, 34)
        assert _eligible(14, 35)

    def test_stays_open_well_past_the_old_cutoff(self):
        # The old window ended at 15:00; these are the slots that were lost.
        for hour, minute in [(15, 0), (15, 15), (15, 30), (15, 55)]:
            assert _eligible(hour, minute), f"{hour}:{minute:02d} should still retry"

    def test_closes_at_the_market_close(self):
        """Never retry into the closed session, where prices go stale."""
        assert not _eligible(16, 0)
        assert not _eligible(16, 30)
        assert not _eligible(17, 30)

    def test_closes_before_the_rack_is_posted(self):
        """A verdict after the rack posts (~6 PM CT) would be useless."""
        assert main.VERDICT_WINDOW_END < (18, 0)

    def test_window_is_materially_wider_than_one_poll_cycle(self):
        start = main.VERDICT_WINDOW_START[0] * 60 + main.VERDICT_WINDOW_START[1]
        end = main.VERDICT_WINDOW_END[0] * 60 + main.VERDICT_WINDOW_END[1]
        minutes = end - start
        assert minutes >= 60, (
            f"only {minutes} minutes of retry; a runner outage of that length "
            f"would still lose the day")
        # At the five-minute poll cadence, how many chances is that?
        assert minutes // 5 >= 12


class TestOutageSurvival:
    """The decisive property: how long an outage can the verdict absorb?"""

    @staticmethod
    def _slots(outage_minutes):
        """Eligible five-minute slots remaining after an outage from 14:35."""
        start = main.VERDICT_WINDOW_START[0] * 60 + main.VERDICT_WINDOW_START[1]
        end = main.VERDICT_WINDOW_END[0] * 60 + main.VERDICT_WINDOW_END[1]
        return sum(1 for t in range(start + outage_minutes, end, 5))

    def test_survives_the_outage_that_actually_happened(self):
        """2026-10-05 lost roughly 25 minutes of runner capacity."""
        assert self._slots(25) > 0, "the real outage would still lose the verdict"

    def test_survives_an_hour_long_outage(self):
        assert self._slots(60) > 0

    def test_a_full_window_outage_is_still_a_miss(self):
        """Honest limit: this widens the window, it does not make it infinite."""
        assert self._slots(85) == 0


def test_late_verdict_is_labelled_as_delayed():
    """The user must be able to tell a catch-up from the normal 2:35 alert."""
    assert main.VERDICT_LATE_AFTER_MINUTES > 0
    assert main.VERDICT_LATE_AFTER_MINUTES <= 30


@pytest.mark.parametrize("hour,minute,expected", [
    (14, 35, False),   # on time
    (14, 44, False),   # within the grace period
    (14, 46, True),    # late
    (15, 30, True),    # very late
])
def test_lateness_threshold(hour, minute, expected):
    late_by = (hour * 60 + minute) - (
        main.VERDICT_WINDOW_START[0] * 60 + main.VERDICT_WINDOW_START[1])
    assert (late_by >= main.VERDICT_LATE_AFTER_MINUTES) is expected


def test_verdict_window_sits_inside_market_hours():
    """A retry slot must never land on a closed market."""
    for hour, minute in [(14, 35), (15, 0), (15, 59)]:
        moment = TZ.localize(datetime(2026, 10, 5, hour, minute))
        assert main.is_market_open(moment), (
            f"{hour}:{minute:02d} CT is inside the verdict window but the market "
            f"is closed, so the run would exit before reaching it")
