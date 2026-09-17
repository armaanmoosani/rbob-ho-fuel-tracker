"""The buy/wait verdict must not be lost to a transient send failure.

``send_email`` used to catch every exception and return None.  ``send_once_today``
then recorded the session as sent regardless, so one SMTP hiccup at 2:35 PM cost
the whole day's verdict: the next five-minute cycle saw the lock and skipped.
"""

from unittest.mock import MagicMock, patch

import pytest

import main


@pytest.fixture
def verdict_data():
    return {
        "RB": {
            "current_price": 2.10, "yesterday_close": 2.00,
            "rack_signal": {
                "text": "Wholesale Gas: HIKE LIKELY (+10.00 c/gal vs prior settle).",
                "risk_text": "Model confidence: 95% that the rack will rise tonight.",
            },
        },
        "HO": {
            "current_price": 2.05, "yesterday_close": 2.00,
            "rack_signal": {"text": "Diesel: NO CLEAR EDGE (+5.00 c/gal).", "risk_text": ""},
        },
    }


class TestLockOnlyAfterDelivery:
    def test_failed_send_does_not_lock_the_session(self, verdict_data):
        state = {}
        with patch("main.send_email", return_value=False), \
             patch("main.load_alert_state", return_value=state), \
             patch("main.save_alert_state") as save:
            delivered = main.send_once_today("VERDICT_1435", "subj", verdict_data,
                                             main.datetime.now(main.TZ), {})
        assert delivered is False
        save.assert_not_called(), "a failed send must leave the day unlocked"

    def test_successful_send_locks_the_session(self, verdict_data):
        state = {}
        with patch("main.send_email", return_value=True), \
             patch("main.load_alert_state", return_value=state), \
             patch("main.save_alert_state") as save:
            delivered = main.send_once_today("VERDICT_1435", "subj", verdict_data,
                                             main.datetime.now(main.TZ), {})
        assert delivered is True
        save.assert_called_once()

    def test_next_cycle_retries_after_a_failure(self, verdict_data):
        """The decisive behaviour: a failure must be followed by another attempt."""
        state = {}
        attempts = []

        def flaky(subject, all_data, now, ctx):
            attempts.append(subject)
            return len(attempts) > 1  # first cycle fails, second succeeds

        now = main.datetime.now(main.TZ)
        with patch("main.send_email", side_effect=flaky), \
             patch("main.load_alert_state", side_effect=lambda: state), \
             patch("main.save_alert_state", side_effect=lambda s: state.update(s)):
            main.send_once_today("VERDICT_1435", "subj", verdict_data, now, {})
            main.send_once_today("VERDICT_1435", "subj", verdict_data, now, {})
            main.send_once_today("VERDICT_1435", "subj", verdict_data, now, {})

        assert len(attempts) == 2, (
            "expected retry after the failure, then a skip once delivered")


class TestPlainTextFallback:
    def test_rich_send_failure_falls_back_and_carries_the_verdict(self, verdict_data):
        sent = {}

        def capture(sender, recipient, payload):
            sent["payload"] = payload

        server = MagicMock()
        server.sendmail.side_effect = capture
        with patch("main.build_html_email", side_effect=RuntimeError("chart build failed")), \
             patch("main.smtplib.SMTP", return_value=server), \
             patch("main.TO_EMAIL", ["buyer@example.com"]), \
             patch("main.GMAIL_USER", "bot@example.com"), \
             patch("main.GMAIL_APP_PASSWORD", "pw"):
            delivered = main.send_email("Final Verdict", verdict_data,
                                        main.datetime.now(main.TZ), {})

        assert delivered is True, "a broken HTML build must not lose the verdict"
        assert "HIKE LIKELY" in sent["payload"]
        assert "95% that the rack will rise" in sent["payload"]

    def test_returns_false_when_nothing_can_be_delivered(self, verdict_data):
        with patch("main.build_html_email", side_effect=RuntimeError("boom")), \
             patch("main.smtplib.SMTP", side_effect=OSError("network down")), \
             patch("main.TO_EMAIL", ["buyer@example.com"]):
            delivered = main.send_email("Final Verdict", verdict_data,
                                        main.datetime.now(main.TZ), {})
        assert delivered is False

    def test_partial_recipient_failure_still_counts_as_delivered(self, verdict_data):
        server = MagicMock()
        server.sendmail.side_effect = [OSError("bad gateway"), None]
        with patch("main.build_html_email", side_effect=RuntimeError("boom")), \
             patch("main.smtplib.SMTP", return_value=server), \
             patch("main.TO_EMAIL", ["sms@vtext.com", "buyer@example.com"]), \
             patch("main.GMAIL_USER", "bot@example.com"), \
             patch("main.GMAIL_APP_PASSWORD", "pw"):
            delivered = main.send_email("Final Verdict", verdict_data,
                                        main.datetime.now(main.TZ), {})
        assert delivered is True, "reaching one of two recipients is still delivery"


def test_successful_rich_send_reports_true(verdict_data):
    server = MagicMock()
    with patch("main.build_html_email", return_value=("<html></html>", {})), \
         patch("main.smtplib.SMTP", return_value=server), \
         patch("main.TO_EMAIL", ["buyer@example.com"]), \
         patch("main.GMAIL_USER", "bot@example.com"), \
         patch("main.GMAIL_APP_PASSWORD", "pw"):
        assert main.send_email("subj", verdict_data, main.datetime.now(main.TZ), {}) is True


class TestProductCoverage:
    """Premium is bought but was never given a verdict."""

    def test_rb_verdict_states_it_covers_premium(self):
        with patch.dict(main.APP_CONFIG, {
            "RB_oos_precision": 0.96, "RB_oos_alerts": 99,
            "RB_oos_window": "2026-03-13..2026-09-16",
        }, clear=False):
            note = main.build_risk_note("RB", "BUY_NOW", 0.92, 5.0)
        assert "premium" in note.lower()

    def test_no_premium_claim_on_diesel_or_on_a_non_alert(self):
        assert "premium" not in main.build_risk_note("HO", "BUY_NOW", 0.92, 5.0).lower()
        assert "premium" not in main.build_risk_note("RB", "NO_EDGE", 0.55, 0.2).lower()


class TestPriceBoundsAreConsistent:
    """A parser stricter than the validator silently drops valid invoices."""

    def test_ingest_bounds_match_the_validator(self):
        import json
        import ingest_prices  # noqa: F401  (import guards against load errors)
        with open("data/config.json") as handle:
            cfg = json.load(handle)
        # validate_data.validate_graves_history accepts [1.00, 10.00].
        assert cfg["PRICE_MIN"] == 1.00
        assert cfg["PRICE_MAX"] == 10.00

    def test_current_diesel_price_is_well_inside_the_ceiling(self):
        import pandas as pd
        history = pd.read_csv("data/graves_history.csv")
        peak = float(history[["rack_u", "rack_p", "rack_d"]].max().max())
        with open("data/config.json") as handle:
            import json
            ceiling = json.load(handle)["PRICE_MAX"]
        assert peak < ceiling * 0.8, (
            f"observed rack peak ${peak:.2f} is within 20% of the ${ceiling:.2f} "
            f"ingest ceiling; raise it before the parser starts rejecting real prices")
