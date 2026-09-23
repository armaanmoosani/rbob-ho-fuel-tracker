"""The live delta must be built from the same source the model was fitted on.

The model is calibrated on settle-to-settle deltas taken from
graves_history.csv.  The live path used to compute its delta against Schwab's
contract-specific ``closePrice`` instead.  Decomposing 74 non-roll live
decisions showed that mismatch was the *entire* live error budget:

    signal price (1:30 PM snapshot) vs recorded settle   robust sd 0.000c
    baseline price                  vs recorded settle   robust sd 0.511c

The 1.2c noise floor built to absorb that is what pinned HO's thresholds at
+/-1.20 instead of the model's own +1.11/-0.74, costing diesel alerts.
"""

import csv
import os
from datetime import date

import pytest
from unittest.mock import patch

import main


def _write_provenance(directory, rows):
    path = os.path.join(directory, "nymex_settlement_provenance.csv")
    columns = ["session_date", "commodity", "settlement_price", "schwab_symbol",
               "yfinance_symbol", "source", "captured_at", "provenance_status"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _write_history(directory, rows):
    path = os.path.join(directory, "graves_history.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "nymex_rb", "nymex_ho", "rack_u", "rack_p", "rack_d"])
        writer.writerows(rows)
    return path


class TestBaselinePreference:
    """Tier 1 beats tier 2 beats nothing."""

    def test_contract_verified_provenance_is_preferred(self, tmp_path):
        _write_provenance(str(tmp_path), [{
            "session_date": "2026-09-16", "commodity": "RB",
            "settlement_price": "3.4830", "schwab_symbol": "/RBV26",
            "provenance_status": "verified"}])
        _write_history(str(tmp_path), [["2026-09-16", "3.9999", "5.0", "2", "3", "2"]])

        with patch("main.DATA_DIR", str(tmp_path)):
            price, source, contract = main.load_baseline_settlement(
                "RB", date(2026, 9, 17), "/RBV26")

        assert price == pytest.approx(3.4830)
        assert source == "settlement_provenance_verified"
        assert contract == "/RBV26"

    def test_provenance_from_a_different_contract_is_refused(self, tmp_path):
        """A recorded settle from another month is not a usable baseline."""
        _write_provenance(str(tmp_path), [{
            "session_date": "2026-09-16", "commodity": "RB",
            "settlement_price": "3.4830", "schwab_symbol": "/RBU26",
            "provenance_status": "verified"}])

        with patch("main.DATA_DIR", str(tmp_path)):
            price, source, _ = main.load_baseline_settlement(
                "RB", date(2026, 9, 17), "/RBV26")
        assert price is None and source is None

    def test_unverified_provenance_is_refused(self, tmp_path):
        _write_provenance(str(tmp_path), [{
            "session_date": "2026-09-16", "commodity": "RB",
            "settlement_price": "3.4830", "schwab_symbol": "/RBV26",
            "provenance_status": "unknown"}])

        with patch("main.DATA_DIR", str(tmp_path)):
            price, source, _ = main.load_baseline_settlement(
                "RB", date(2026, 9, 17), "/RBV26")
        assert price is None and source is None

    def test_falls_back_to_history_when_the_contract_cannot_have_changed(self, tmp_path):
        _write_history(str(tmp_path), [["2026-09-16", "3.4830", "5.2453", "2", "3", "2"]])
        with patch("main.DATA_DIR", str(tmp_path)), \
             patch("main.is_contract_roll_day", return_value=False):
            price, source, _ = main.load_baseline_settlement(
                "RB", date(2026, 9, 17), "/RBV26")
        assert price == pytest.approx(3.4830)
        assert source == "graves_history_same_contract"

    def test_history_is_refused_on_a_roll_session(self, tmp_path):
        """On a roll the stored settle may be the previous month's contract."""
        _write_history(str(tmp_path), [["2026-09-16", "3.4830", "5.2453", "2", "3", "2"]])
        with patch("main.DATA_DIR", str(tmp_path)), \
             patch("main.is_contract_roll_day", return_value=True):
            price, source, _ = main.load_baseline_settlement(
                "RB", date(2026, 9, 17), "/RBV26")
        assert price is None and source is None

    def test_missing_previous_session_yields_nothing(self, tmp_path):
        _write_history(str(tmp_path), [["2026-01-02", "3.0", "4.0", "2", "3", "2"]])
        with patch("main.DATA_DIR", str(tmp_path)), \
             patch("main.is_contract_roll_day", return_value=False):
            price, source, _ = main.load_baseline_settlement(
                "RB", date(2026, 9, 17), "/RBV26")
        assert price is None and source is None


class TestThresholdWidening:
    """A mismatched baseline must not get the tightened thresholds."""

    @staticmethod
    def _signal(baseline_source, move_cents, tmp_path):
        data = {
            "current_price": 2.00 + move_cents / 100.0,
            "yesterday_close": 2.00,
            "open_price": 2.00, "high_price": 2.2, "low_price": 1.9,
            "daily_pct": 0.0, "five_day_high": 2.2, "five_day_low": 1.9,
            "thirty_day_avg": 2.0, "schwab_symbol": "/HOV26",
            "baseline_source": baseline_source,
        }
        with patch("main.DATA_DIR", str(tmp_path)), \
             patch("main.load_settlement_snapshot", return_value=None), \
             patch("main.load_alert_state", return_value={}), \
             patch("main.is_contract_roll_day", return_value=False), \
             patch("main.get_decoupling_warning", return_value=""):
            return main.build_rack_signal("HO", data, main.datetime.now(main.TZ))

    def test_matched_baseline_uses_the_published_threshold(self, tmp_path):
        published = main.APP_CONFIG["HO_DROP_THRESHOLD_CENTS"]
        signal = self._signal("settlement_provenance_verified",
                              published - 0.01, tmp_path)
        assert signal["action"] == "WAIT"

    def test_mismatched_baseline_is_held_to_the_fallback_floor(self, tmp_path):
        """The decisive case: a move inside the mismatch budget must not fire.

        HO's published drop threshold is -0.74c, well inside the 1.04c p95 of
        the baseline source mismatch.  With a Schwab baseline that move is not
        distinguishable from the measurement difference.
        """
        fallback = main.APP_CONFIG.get("FALLBACK_NOISE_FLOOR_CENTS", 1.2)
        published = main.APP_CONFIG["HO_DROP_THRESHOLD_CENTS"]
        if abs(published) >= fallback:
            pytest.skip("published threshold already clears the fallback floor")

        move = published - 0.01          # would trigger on a matched baseline
        assert abs(move) < fallback      # but sits inside the mismatch budget

        matched = self._signal("settlement_provenance_verified", move, tmp_path)
        mismatched = self._signal("schwab_close_price", move, tmp_path)

        assert matched["action"] == "WAIT"
        assert mismatched["action"] != "WAIT", (
            "a Schwab baseline must not fire a threshold the source mismatch "
            "alone could produce")

    def test_a_decisive_move_still_fires_on_a_mismatched_baseline(self, tmp_path):
        fallback = main.APP_CONFIG.get("FALLBACK_NOISE_FLOOR_CENTS", 1.2)
        signal = self._signal("schwab_close_price", -(fallback + 5.0), tmp_path)
        assert signal["action"] == "WAIT"

        with open(tmp_path / "prediction_log.csv", newline="") as handle:
            row = next(csv.DictReader(handle))
        assert float(row["threshold_used"]) == pytest.approx(-fallback)
        assert float(row["drop_threshold_used"]) == pytest.approx(-fallback)
        assert float(row["lean_drop_threshold_used"]) == pytest.approx(-fallback)

    def test_lean_band_never_inverts_after_widening(self, tmp_path):
        for source in ("settlement_provenance_verified", "schwab_close_price", "unknown"):
            signal = self._signal(source, 0.0, tmp_path)
            assert signal["action"] == "NO_EDGE"


def test_calibrated_sources_are_exactly_the_calibration_sources():
    """Guards against a new baseline tier silently earning tight thresholds."""
    assert main.CALIBRATED_BASELINE_SOURCES == frozenset({
        "settlement_provenance_verified", "graves_history_same_contract"})
    for source in ("schwab_close_price", "graves_history_unverified",
                   "yfinance_contract_daily", "yfinance_continuous_unverified",
                   "unknown"):
        assert source not in main.CALIBRATED_BASELINE_SOURCES


def test_the_two_floors_are_ordered():
    import backtest
    assert (backtest.DEFAULTS["SNAPSHOT_NOISE_FLOOR_CENTS"]
            < backtest.DEFAULTS["FALLBACK_NOISE_FLOOR_CENTS"]), (
        "the matched-source floor must be the tighter of the two")


def test_recorded_provenance_matches_graves_history_exactly():
    """The premise of the whole change, asserted on the real files.

    If these two ever diverge, the provenance record is no longer the value the
    model was fitted on and tier 1 must not be trusted.
    """
    import pandas as pd
    provenance = pd.read_csv("data/nymex_settlement_provenance.csv")
    history = pd.read_csv("data/graves_history.csv").set_index("date")
    column = {"RB": "nymex_rb", "HO": "nymex_ho"}

    checked = 0
    for _, row in provenance.iterrows():
        session, commodity = row["session_date"], row["commodity"]
        if session not in history.index:
            continue
        recorded = history.loc[session, column[commodity]]
        if pd.isna(recorded):
            continue
        assert abs(float(row["settlement_price"]) - float(recorded)) < 1e-6, (
            f"{session} {commodity}: provenance {row['settlement_price']} != "
            f"history {recorded}")
        checked += 1
    assert checked >= 50, f"only {checked} sessions cross-checked"
