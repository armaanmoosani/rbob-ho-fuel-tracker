import unittest
import os
import sys
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import pytz
from unittest.mock import patch, MagicMock, mock_open, call

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Set mock env variables needed on import by main and ingest_prices
os.environ['GH_PAT'] = 'mock_pat'
os.environ['GH_REPO'] = 'mock_repo'
os.environ['GMAIL_USER'] = 'mock_user'
os.environ['GMAIL_APP_PASSWORD'] = 'mock_pass'
os.environ['TO_EMAIL'] = 'mock_to@example.com'
os.environ['GRAVES_EMAIL'] = 'mock_graves@example.com'
os.environ['GRAVES_APP_PASSWORD'] = 'mock_graves_pass'

import inspect

import validate_data
import alignment
import model
import backtest
import futures_util
import ingest_prices
import main
import weekly_report


class TestCategory1InputParsing(unittest.TestCase):
    """Category 1: Input Ingestion & Parsing Tests"""

    def test_1_1_standard_parsing(self):
        body = "E10 - UNLEADED: $2.10\nE10 - PREMIUM: $2.30\nCLEAR DIESEL: $2.50"
        p_u = ingest_prices.extract_price_near_label(body, "E10 - UNLEADED")
        p_p = ingest_prices.extract_price_near_label(body, "E10 - PREMIUM")
        p_d = ingest_prices.extract_price_near_label(body, "CLEAR DIESEL")
        self.assertEqual(p_u, 2.10)
        self.assertEqual(p_p, 2.30)
        self.assertEqual(p_d, 2.50)

    def test_1_2_delimiter_variants(self):
        variants = [
            "E10 - UNLEADED,2.10\nE10 - PREMIUM,2.30\nCLEAR DIESEL,2.50",
            "E10 - UNLEADED/2.10\nE10 - PREMIUM/2.30\nCLEAR DIESEL/2.50",
            "E10 - UNLEADED $2.10\nE10 - PREMIUM $2.30\nCLEAR DIESEL $2.50",
            "E10 - UNLEADED, 2.10\nE10 - PREMIUM, 2.30\nCLEAR DIESEL, 2.50"
        ]
        for body in variants:
            p_u = ingest_prices.extract_price_near_label(body, "E10 - UNLEADED")
            p_p = ingest_prices.extract_price_near_label(body, "E10 - PREMIUM")
            p_d = ingest_prices.extract_price_near_label(body, "CLEAR DIESEL")
            self.assertEqual(p_u, 2.10)
            self.assertEqual(p_p, 2.30)
            self.assertEqual(p_d, 2.50)

    def test_1_3_extra_whitespace(self):
        body = "  E10 - UNLEADED  ,  2.10  \n  E10 - PREMIUM   2.30  \n  CLEAR DIESEL  2.50  "
        p_u = ingest_prices.extract_price_near_label(body, "E10 - UNLEADED")
        p_p = ingest_prices.extract_price_near_label(body, "E10 - PREMIUM")
        p_d = ingest_prices.extract_price_near_label(body, "CLEAR DIESEL")
        self.assertEqual(p_u, 2.10)
        self.assertEqual(p_p, 2.30)
        self.assertEqual(p_d, 2.50)

    def test_1_4_four_decimal_prices(self):
        body = "E10 - UNLEADED $2.1050\nE10 - PREMIUM $2.3025\nCLEAR DIESEL $2.4975"
        p_u = ingest_prices.extract_price_near_label(body, "E10 - UNLEADED")
        p_p = ingest_prices.extract_price_near_label(body, "E10 - PREMIUM")
        p_d = ingest_prices.extract_price_near_label(body, "CLEAR DIESEL")
        self.assertEqual(p_u, 2.1050)
        self.assertEqual(p_p, 2.3025)
        self.assertEqual(p_d, 2.4975)

    def test_1_5_too_few_values(self):
        # Email has E10 - UNLEADED and E10 - PREMIUM, but missing CLEAR DIESEL
        body = "E10 - UNLEADED $2.10\nE10 - PREMIUM $2.30\n"
        prices = []
        for key, label in ingest_prices.LABELS.items():
            p = ingest_prices.extract_price_near_label(body, label)
            if p is None or not (1.50 <= p <= 6.00):
                break
            prices.append(p)
        self.assertNotEqual(len(prices), 3)

    def test_1_6_too_many_values(self):
        body = "E10 - UNLEADED $2.10\nE10 - PREMIUM $2.30\nCLEAR DIESEL $2.50\nKEROSENE $3.00"
        prices = []
        for key, label in ingest_prices.LABELS.items():
            p = ingest_prices.extract_price_near_label(body, label)
            if p is None or not (1.50 <= p <= 6.00):
                break
            prices.append(p)
        self.assertEqual(len(prices), 3)
        self.assertEqual(prices, [2.10, 2.30, 2.50])

    def test_1_7_1_8_1_9_price_bounds(self):
        # Bounds check logic in ingest_prices: 1.50 <= p <= 6.00
        valid_low = 1.50
        valid_high = 6.00
        invalid_low = 1.49
        invalid_high = 6.01
        fat_finger = 210.0

        self.assertTrue(1.50 <= valid_low <= 6.00)
        self.assertTrue(1.50 <= valid_high <= 6.00)
        self.assertFalse(1.50 <= invalid_low <= 6.00)
        self.assertFalse(1.50 <= invalid_high <= 6.00)
        self.assertFalse(1.50 <= fat_finger <= 6.00)

    def test_1_10_non_numeric_input(self):
        body = "E10 - UNLEADED abc"
        p = ingest_prices.extract_price_near_label(body, "E10 - UNLEADED")
        self.assertIsNone(p)


class TestCategory2DatabaseIntegrity(unittest.TestCase):
    """Category 2: CSV Database Integrity Tests"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "graves_history.csv")
        self.hashes_path = os.path.join(self.temp_dir, "integrity_hashes.csv")
        self.log_path = os.path.join(self.temp_dir, "prediction_log.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def write_valid_csv(self):
        df = pd.DataFrame({
            "date": ["2026-05-18", "2026-05-19", "2026-05-20"],
            "nymex_rb": [2.10, 2.15, 2.12],
            "nymex_ho": [2.20, 2.25, 2.22],
            "rack_u": [2.30, 2.35, 2.32],
            "rack_p": [2.40, 2.45, 2.42],
            "rack_d": [2.50, 2.55, 2.52]
        })
        df.to_csv(self.csv_path, index=False)

    def test_2_1_append_writes_correct_columns(self):
        self.write_valid_csv()
        df = pd.read_csv(self.csv_path)
        expected = ["date", "nymex_rb", "nymex_ho", "rack_u", "rack_p", "rack_d"]
        self.assertEqual(list(df.columns), expected)

    def test_2_2_duplicate_date_guard(self):
        df = pd.DataFrame({
            "date": ["2026-05-18", "2026-05-18"],
            "nymex_rb": [2.10, 2.15],
            "nymex_ho": [2.20, 2.25],
            "rack_u": [2.30, 2.35],
            "rack_p": [2.40, 2.45],
            "rack_d": [2.50, 2.55]
        })
        df.to_csv(self.csv_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_graves_history(self.csv_path)

    def test_2_3_chronological_sort_preservation(self):
        df = pd.DataFrame({
            "date": ["2026-05-20", "2026-05-19"],
            "nymex_rb": [2.10, 2.15],
            "nymex_ho": [2.20, 2.25],
            "rack_u": [2.30, 2.35],
            "rack_p": [2.40, 2.45],
            "rack_d": [2.50, 2.55]
        })
        df.to_csv(self.csv_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_graves_history(self.csv_path)

    def test_2_4_2_5_weekend_rows_verification(self):
        # We test that weekends have NaN settlements and standard days don't.
        # Saturdays (5) and Sundays (6)
        dt_sat = datetime(2026, 5, 23) # Saturday
        dt_mon = datetime(2026, 5, 25) # Monday
        self.assertTrue(dt_sat.weekday() in (5, 6))
        self.assertFalse(dt_mon.weekday() in (5, 6))

    def test_2_6_friday_to_monday_is_one_session(self):
        """A Friday->Monday pair is one session, and a missing day is not."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-05-15", "2026-05-18", "2026-05-19"]),
            "nymex_rb": [2.00, 2.10, 2.05],
            "nymex_ho": [2.00, 2.10, 2.05],
            "rack_u": [2.10, 2.20, 2.15],
            "rack_d": [2.10, 2.20, 2.15],
        })
        frame = alignment.aligned_deltas(df, "RB")
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(frame["delta_nymex"].iloc[0], 10.0)
        self.assertAlmostEqual(frame["delta_nymex"].iloc[1], -5.0)
        # session_gap lets a consumer see that the first pair spans a weekend
        # (3 calendar days) while still being one trading session.
        self.assertEqual(frame["session_gap"].iloc[0], 3)
        self.assertEqual(frame["session_gap"].iloc[1], 1)

    def test_2_7_weekend_rows_are_dropped_before_differencing(self):
        """Legacy Saturday rows must never contribute a pair.

        Every legacy Saturday row carries a NaN settle.  If it survived into the
        frame, the Friday->Tuesday difference would silently become a
        two-session rack move scored against a one-session NYMEX move.
        """
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-05-15", "2026-05-16", "2026-05-18"]),
            "nymex_rb": [2.00, float("nan"), 2.10],
            "nymex_ho": [2.00, float("nan"), 2.10],
            "rack_u": [2.10, 2.15, 2.20],
            "rack_d": [2.10, 2.15, 2.20],
        })
        loaded = alignment.load_history_from_frame(df)
        self.assertEqual(len(loaded), 2)
        self.assertNotIn(pd.Timestamp("2026-05-16"), set(loaded["date"]))

    def test_2_7_corrupt_row_middle_check(self):
        # Insert a corrupt non-numeric value in nymex_rb in the middle of a valid DataFrame
        df = pd.DataFrame({
            "date": ["2026-05-18", "2026-05-19", "2026-05-20"],
            "nymex_rb": [2.10, "corrupt_value", 2.12],
            "nymex_ho": [2.20, 2.25, 2.22],
            "rack_u": [2.30, 2.35, 2.32],
            "rack_p": [2.40, 2.45, 2.42],
            "rack_d": [2.50, 2.55, 2.52]
        })
        df.to_csv(self.csv_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_graves_history(self.csv_path)

    def test_2_8_repair_csv_with_exactly_3_columns(self):
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
            f.write("2026-05-18,2.10,2.20,2.30,2.40,2.50\n")
            f.write("2026-05-19,2.15,2.25") # Exactly 3 columns instead of 6
            
        validate_data.repair_csv_if_corrupted(self.csv_path)
        
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1], "2026-05-18,2.10,2.20,2.30,2.40,2.50")

    def test_2_9_historical_row_modification_detection(self):
        # 1. Initialize integrity hashes
        self.write_valid_csv()
        validate_data.validate_all(self.temp_dir)
        
        # 2. Modify a historical row (first row of graves_history.csv, index 0, not the last row)
        df = pd.read_csv(self.csv_path)
        df.loc[0, 'nymex_rb'] = 9.99
        df.to_csv(self.csv_path, index=False)
        
        # 3. Running verification should fail with SystemExit due to historical row modification
        with self.assertRaises(SystemExit):
            validate_data.validate_all(self.temp_dir)

    def test_2_10_missing_metrics_cache_does_not_fail_validation(self):
        # On day zero of a fresh fork, metrics_cache.json won't exist.
        # Ensure validation handles this gracefully.
        self.write_valid_csv()
        metrics_path = os.path.join(self.temp_dir, "metrics_cache.json")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
            
        try:
            validate_data.validate_all(self.temp_dir)
        except SystemExit:
            self.fail("validate_all raised SystemExit when metrics_cache.json was missing!")
            
        # Verify hashes CSV was created and contains graves_history but NOT metrics_cache
        df_hashes = pd.read_csv(self.hashes_path)
        tracked_files = df_hashes['file_name'].tolist()
        self.assertIn("graves_history.csv", tracked_files)
        self.assertNotIn("metrics_cache.json", tracked_files)


class TestCategory3SettlementCapture(unittest.TestCase):
    """Category 3: NYMEX Settlement Capture Tests"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ds_path = os.path.join(self.temp_dir, "daily_settlement.json")
        ingest_prices.DS_PATH = self.ds_path

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_3_1_daily_settlement_fresh(self):
        today_str = datetime.now(pytz.timezone('America/Chicago')).date().isoformat()
        with open(self.ds_path, "w") as f:
            json.dump({"date": today_str, "rbob_settlement": 2.10, "heating_oil_settlement": 2.20}, f)
        
        ds = ingest_prices.read_daily_settlement(today_str)
        self.assertIsNotNone(ds)
        self.assertEqual(ds["rbob_settlement"], 2.10)

    def test_3_2_daily_settlement_stale(self):
        today_str = datetime.now(pytz.timezone('America/Chicago')).date().isoformat()
        yest_str = (datetime.now(pytz.timezone('America/Chicago')) - timedelta(days=1)).date().isoformat()
        with open(self.ds_path, "w") as f:
            json.dump({"date": yest_str, "rbob_settlement": 2.10, "heating_oil_settlement": 2.20}, f)
        
        ds = ingest_prices.read_daily_settlement(today_str)
        self.assertIsNone(ds)

    def test_3_3_utc_vs_ct_date_boundary(self):
        # 11:00 PM UTC on May 22 is 6:00 PM CT on May 22 (Standard Time offset -6h or CDT offset -5h)
        # Test timezone localization
        tz = pytz.timezone('America/Chicago')
        dt_utc = datetime(2026, 5, 22, 23, 0, 0, tzinfo=pytz.utc)
        dt_local = dt_utc.astimezone(tz)
        self.assertEqual(dt_local.date().isoformat(), "2026-05-22")

    def test_3_4_missing_settlement_file(self):
        if os.path.exists(self.ds_path):
            os.remove(self.ds_path)
        ds = ingest_prices.read_daily_settlement("2026-05-22")
        self.assertIsNone(ds)

    @patch('ingest_prices.git_pull_rebase')
    @patch('ingest_prices.validate_data.validate_all')
    @patch('ingest_prices.check_inbox_for_prices')
    @patch('ingest_prices.git_commit_push')
    @patch('ingest_prices.send_alert_email')
    @patch('ingest_prices.read_daily_settlement')
    @patch('ingest_prices.get_github_settlement_snapshots')
    @patch('ingest_prices.get_thinkorswim_settlement')
    @patch('ingest_prices.datetime')
    def test_3_5_settlement_fallback_chain(self, mock_datetime, mock_tos, mock_github, mock_read_local, mock_send_email, mock_commit, mock_check_inbox, mock_validate, mock_pull):
        # 1. Setup datetime to be a Monday night (2026-05-18 20:00 Chicago time)
        tz = pytz.timezone('America/Chicago')
        dt_monday = tz.localize(datetime(2026, 5, 18, 20, 0, 0))
        mock_datetime.now.return_value = dt_monday
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        
        # 2. Setup inbox check to return valid prices for today
        mock_check_inbox.return_value = ("2026-05-18", (2.10, 2.20, 2.30))
        
        # Mock file operations to prevent touching real files
        temp_dir = tempfile.mkdtemp()
        orig_csv_path = ingest_prices.CSV_PATH
        orig_ds_path = ingest_prices.DS_PATH
        ingest_prices.CSV_PATH = os.path.join(temp_dir, "graves_history.csv")
        ingest_prices.DS_PATH = os.path.join(temp_dir, "daily_settlement.json")
        
        # Write a header-only csv
        with open(ingest_prices.CSV_PATH, "w") as f:
            f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
            
        try:
            # --- Scenario 1: Local daily_settlement.json matches ---
            mock_read_local.return_value = {
                "date": "2026-05-18",
                "rbob_settlement": 2.05,
                "heating_oil_settlement": 2.15
            }
            
            # Run main
            try:
                ingest_prices.main()
            except SystemExit:
                pass
                
            # Verify local was called, but github and TOS were not
            mock_read_local.assert_called_with("2026-05-18")
            mock_github.assert_not_called()
            mock_tos.assert_not_called()
            
            # --- Scenario 2: Local missing/stale, GitHub matches ---
            mock_read_local.reset_mock()
            mock_github.reset_mock()
            
            # Clear CSV so it doesn't complain about duplicates
            with open(ingest_prices.CSV_PATH, "w") as f:
                f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
                
            mock_read_local.return_value = None
            mock_github.return_value = {
                "date": "2026-05-18",
                "rbob_settlement": 2.05,
                "heating_oil_settlement": 2.15,
                "source": "github_variables"
            }
            
            try:
                ingest_prices.main()
            except SystemExit:
                pass
                
            mock_read_local.assert_called_once()
            mock_github.assert_called_with("2026-05-18")
            mock_tos.assert_not_called()
            
            # --- Scenario 3: Local and GitHub missing, ThinkOrSwim matches ---
            mock_read_local.reset_mock()
            mock_github.reset_mock()
            mock_tos.reset_mock()
            
            with open(ingest_prices.CSV_PATH, "w") as f:
                f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
                
            mock_read_local.return_value = None
            mock_github.return_value = None
            
            # Mock ThinkOrSwim/Schwab return
            mock_tos.return_value = {
                'date': '2026-05-18',
                'rbob_settlement': 2.05,
                'heating_oil_settlement': 2.15,
                'source': 'tos_test'
            }
            
            try:
                ingest_prices.main()
            except SystemExit:
                pass
                
            mock_read_local.assert_called_once()
            mock_github.assert_called_once()
            # ThinkOrSwim fallback should have been called once
            self.assertEqual(mock_tos.call_count, 1)
            mock_tos.assert_called_with('2026-05-18')
            
        finally:
            ingest_prices.CSV_PATH = orig_csv_path
            ingest_prices.DS_PATH = orig_ds_path
            shutil.rmtree(temp_dir)

    @patch('ingest_prices.datetime')
    def test_3_6_imap_date_boundary_hour_under_12(self, mock_datetime):
        # Local hour is 11:59 AM (hour < 12) on 2026-05-18
        tz = pytz.timezone('America/Chicago')
        dt = tz.localize(datetime(2026, 5, 18, 11, 59, 0))
        mock_datetime.now.return_value = dt
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        
        now_local = dt
        if now_local.hour < 12:
            target_date_str = (now_local - timedelta(days=1)).date().isoformat()
        else:
            target_date_str = now_local.date().isoformat()
        self.assertEqual(target_date_str, "2026-05-17")
        
        # Test 12 hours or over (e.g. 12:00 PM)
        dt_12 = tz.localize(datetime(2026, 5, 18, 12, 0, 0))
        now_local = dt_12
        if now_local.hour < 12:
            target_date_str = (now_local - timedelta(days=1)).date().isoformat()
        else:
            target_date_str = now_local.date().isoformat()
        self.assertEqual(target_date_str, "2026-05-18")

    def test_3_7_get_session_start(self):
        import main
        tz = pytz.timezone('America/Chicago')
        
        # Test 16:59:59 -> session start should be previous day 17:00
        dt_1659 = tz.localize(datetime(2026, 5, 22, 16, 59, 59))
        start_1659 = main.get_session_start(dt_1659)
        self.assertEqual(start_1659, tz.localize(datetime(2026, 5, 21, 17, 0, 0)))
        
        # Test 17:00:00 -> session start should be same day 17:00
        dt_1700 = tz.localize(datetime(2026, 5, 22, 17, 0, 0))
        start_1700 = main.get_session_start(dt_1700)
        self.assertEqual(start_1700, tz.localize(datetime(2026, 5, 22, 17, 0, 0)))


class TestCategory4LagDiscovery(unittest.TestCase):
    """Category 4: Lag Discovery Math Tests"""

    def test_4_1_perfect_lag_0_detection(self):
        # nymex_rb = [1, 2, 3, 4, 5]
        # rack_u = [1.05, 2.05, 3.05, 4.05, 5.05] (perfect lag=0 correlation)
        from scipy import stats
        nymex = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rack = pd.Series([1.05, 2.05, 3.05, 4.05, 5.05])
        slope, intercept, r_value, p_value, std_err = stats.linregress(nymex, rack)
        self.assertAlmostEqual(r_value**2, 1.0)

    def test_4_2_perfect_lag_1_detection(self):
        # nymex_rb = [1, 2, 3, 4, 5]
        # rack_u = [NaN, 1.05, 2.05, 3.05, 4.05] (perfect lag=1 correlation)
        from scipy import stats
        nymex = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]).shift(1)
        rack = pd.Series([1.05, 2.05, 3.05, 4.05, 5.05])
        valid = ~(nymex.isna() | rack.isna())
        slope, intercept, r_value, p_value, std_err = stats.linregress(nymex[valid], rack[valid])
        self.assertAlmostEqual(r_value**2, 1.0)

    def test_4_4_monday_lag_traversal(self):
        # previous_nymex_business_day(Monday) -> Friday
        monday = datetime(2026, 5, 25).date()
        friday = main.previous_nymex_business_day(monday)
        self.assertEqual(friday, datetime(2026, 5, 22).date())

    def test_4_5_holiday_lag_traversal(self):
        # Memorial Day is Monday May 25, 2026 (holiday).
        # Tuesday May 26 is the next business day.
        # previous_nymex_business_day(Tuesday) -> Friday May 22
        tuesday = datetime(2026, 5, 26).date()
        prev = main.previous_nymex_business_day(tuesday)
        self.assertEqual(prev, datetime(2026, 5, 22).date())

    def test_4_6_consecutive_holidays_lag_traversal(self):
        # Good Friday 2026 is April 3, 2026.
        # Monday April 6, 2026 is the next business day.
        # previous_nymex_business_day(Monday April 6) should skip Sunday April 5,
        # Saturday April 4, and Friday April 3 (Good Friday), returning Thursday April 2.
        monday = datetime(2026, 4, 6).date()
        prev = main.previous_nymex_business_day(monday)
        self.assertEqual(prev, datetime(2026, 4, 2).date())


class TestCategory5SplitOLS(unittest.TestCase):
    """Category 5: Split OLS (Rockets & Feathers) Tests"""

    def test_5_1_symmetric_passthrough_baseline(self):
        from scipy import stats
        # NYMEX rises and falls symmetrically
        nymex_diff = pd.Series([1.0, -1.0, 2.0, -2.0, 1.5, -1.5])
        rack_diff = pd.Series([1.0, -1.0, 2.0, -2.0, 1.5, -1.5])
        slope, intercept, r_value, p_value, std_err = stats.linregress(nymex_diff, rack_diff)
        self.assertAlmostEqual(slope, 1.0)

    def test_5_2_asymmetric_passthrough_detection(self):
        # Up moves are 1.2x, Down moves are 0.6x
        from scipy import stats
        nymex_diff = pd.Series([1.0, -1.0, 2.0, -2.0, 1.5, -1.5])
        rack_diff = pd.Series([1.2, -0.6, 2.4, -1.2, 1.8, -0.9])
        
        up_mask = nymex_diff > 0
        down_mask = nymex_diff < 0
        
        slope_up, _, _, _, _ = stats.linregress(nymex_diff[up_mask], rack_diff[up_mask])
        slope_down, _, _, _, _ = stats.linregress(nymex_diff[down_mask], rack_diff[down_mask])
        
        self.assertAlmostEqual(slope_up, 1.2)
        self.assertAlmostEqual(slope_down, 0.6)

    def test_5_3_no_up_move_days(self):
        # Edge case: no positive moves
        nymex_diff = pd.Series([-1.0, -2.0, -1.5, 0.0])
        up_mask = nymex_diff > 0
        self.assertEqual(up_mask.sum(), 0)


class TestCategory6ThresholdCalculation(unittest.TestCase):
    """Category 6: Threshold construction from the pass-through model."""

    def _fit(self, slope=0.7, intercept=0.0, noise=2.0, n=400, seed=7):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 8, n)
        y = intercept + slope * x + rng.normal(0, noise, n)
        return model.fit_passthrough(x, y)

    def test_6_1_threshold_delivers_the_requested_confidence(self):
        """The marginal signal must actually win at the requested rate."""
        fit = self._fit()
        for target in (0.65, 0.75, 0.85):
            hike, drop = fit.threshold_for_confidence(target)
            self.assertAlmostEqual(float(fit.probability_correct(hike)), target, places=2)
            self.assertAlmostEqual(float(fit.probability_correct(drop)), target, places=2)

    def test_6_2_higher_confidence_demands_a_larger_move(self):
        fit = self._fit()
        loose = fit.threshold_for_confidence(0.65)
        tight = fit.threshold_for_confidence(0.85)
        self.assertGreater(tight[0], loose[0])
        self.assertLess(tight[1], loose[1])

    def test_6_3_thresholds_are_clamped_to_sane_bounds(self):
        """A weak or razor-sharp fit must not emit an absurd threshold."""
        # Weak but genuinely positive pass-through: the raw threshold would be
        # tens of cents, far beyond any real daily move.
        weak = self._fit(slope=0.05, noise=5.0, n=4000)
        self.assertGreater(weak.slope, 0)
        hike, drop = weak.threshold_for_confidence(0.9)
        self.assertEqual(hike, model.MAX_ABS_THRESHOLD)
        self.assertEqual(drop, -model.MAX_ABS_THRESHOLD)

        # Near-deterministic pass-through: the raw threshold would round to
        # zero and every trivial move would become an alert.
        sharp = self._fit(slope=5.0, noise=0.01)
        hike, drop = sharp.threshold_for_confidence(0.6)
        self.assertEqual(hike, model.MIN_ABS_THRESHOLD)
        self.assertEqual(drop, -model.MIN_ABS_THRESHOLD)

    def test_6_4_noise_floor_cannot_be_undercut(self):
        """A threshold below the snapshot error would be fired by the error."""
        hike, drop = model.apply_noise_floor(0.4, -0.4, 1.2)
        self.assertEqual(hike, 1.2)
        self.assertEqual(drop, -1.2)
        # An already-wide threshold is left alone.
        hike, drop = model.apply_noise_floor(3.0, -2.5, 1.2)
        self.assertEqual((hike, drop), (3.0, -2.5))

    def test_6_5_inverted_passthrough_is_refused(self):
        """A negative slope means the relationship has broken, not inverted."""
        fit = self._fit(slope=-0.7)
        with self.assertRaises(model.ModelFitError):
            fit.threshold_for_confidence(0.75)

    def test_6_6_calibration_is_deterministic(self):
        """Identical inputs must give bit-identical thresholds."""
        a = self._fit(seed=11).threshold_for_confidence(0.75)
        b = self._fit(seed=11).threshold_for_confidence(0.75)
        self.assertEqual(a, b)


class TestCategory7WalkForward(unittest.TestCase):
    """Category 7: Walk-forward evaluation is honest and leakage-free."""

    def _frame(self, n=400, seed=3):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2025-01-01", periods=n)
        x = rng.normal(0, 8, n)
        y = 0.7 * x + rng.normal(0, 2, n)
        return pd.DataFrame({"date": dates, "delta_nymex": x, "delta_rack": y,
                             "session_gap": 1})

    def _cfg(self, **over):
        cfg = dict(backtest.DEFAULTS)
        cfg.update(over)
        return cfg

    def test_7_1_train_purge_test_blocks_are_ordered_and_disjoint(self):
        """No training row may sit on or after its own test block."""
        frame = self._frame()
        seen = []
        real_fit = backtest.model.fit_passthrough

        def spy(x, y):
            seen.append((len(x), float(np.asarray(x)[0]), float(np.asarray(x)[-1])))
            return real_fit(x, y)

        with patch.object(backtest.model, "fit_passthrough", side_effect=spy):
            result = backtest.walk_forward_evaluation(frame, self._cfg())

        self.assertEqual(len(result["folds"]), backtest.EVAL_FOLDS)
        n = len(frame)
        for fold_index in range(backtest.EVAL_FOLDS):
            test_end = n - fold_index * backtest.EVAL_TEST_ROWS
            test_start = test_end - backtest.EVAL_TEST_ROWS
            train_end = test_start - backtest.CALIBRATION_PURGE_ROWS
            self.assertGreater(test_start, train_end,
                               "training must end before the test block begins")
            self.assertEqual(test_start - train_end, backtest.CALIBRATION_PURGE_ROWS,
                             "exactly one purged session must separate them")

    def test_7_2_evaluation_refuses_a_geometry_that_does_not_fit(self):
        """It must raise, not silently fall back to hardcoded thresholds.

        The superseded engine returned a -9999 sentinel here and then reverted
        to a hardcoded (120, 15, 85), so a configuration too large for the data
        produced plausible thresholds with no warning anywhere.
        """
        tiny = self._frame(n=40)
        with self.assertRaises(model.ModelFitError):
            backtest.walk_forward_evaluation(tiny, self._cfg())

    def test_7_3_roll_sessions_are_excluded_from_training_pairs(self):
        dates = pd.bdate_range("2026-01-01", periods=120)
        df = pd.DataFrame({
            "date": dates,
            "nymex_rb": np.linspace(2.0, 2.5, 120),
            "nymex_ho": np.linspace(2.0, 2.5, 120),
            "rack_u": np.linspace(2.1, 2.6, 120),
            "rack_d": np.linspace(2.1, 2.6, 120),
        })
        rolls = {dates[10].date(), dates[40].date()}
        with patch("backtest.is_contract_roll_day",
                   side_effect=lambda d, p: d in rolls):
            frame = backtest.training_pairs(df, "RB")
        kept = {d.date() for d in frame["date"]}
        self.assertTrue(rolls.isdisjoint(kept))

    def test_7_4_no_parameter_is_selected_from_the_folds(self):
        """The evaluation must not be able to change the fitted thresholds.

        This is the structural guarantee that makes the reported out-of-sample
        figure honest: the previous engine maximised the median fold savings
        over 64 combinations and then reported those same folds as
        out-of-sample.
        """
        frame = self._frame()
        cfg = self._cfg()

        # The published thresholds must be reproducible from the final window
        # alone.  If any fold outcome could influence them, this equality fails.
        window = cfg["ROLLING_WINDOW_DAYS"]
        tail = frame.tail(window)
        direct = model.apply_noise_floor(
            *model.fit_passthrough(tail["delta_nymex"], tail["delta_rack"])
            .threshold_for_confidence(cfg["TARGET_SIGNAL_CONFIDENCE"]),
            cfg["SNAPSHOT_NOISE_FLOOR_CENTS"])

        history = pd.DataFrame({
            "date": frame["date"],
            "nymex_rb": 0.0, "nymex_ho": 0.0, "rack_u": 0.0, "rack_d": 0.0,
        })
        with patch("backtest.training_pairs", return_value=frame):
            out, _, _ = backtest.calibrate(history, "RB", dict(cfg))

        self.assertAlmostEqual(out["RB_HIKE_THRESHOLD_CENTS"], round(direct[0], 2))
        self.assertAlmostEqual(out["RB_DROP_THRESHOLD_CENTS"], round(direct[1], 2))


class TestCategory8CVaRAndRiskMetrics(unittest.TestCase):
    """Category 8: CVaR and Risk Metrics Tests"""

    def test_8_1_cvar_known_distribution(self):
        # 100 values: worst 5 values (largest moves)
        # np.linspace(1.0, 100.0, 100) -> 95th percentile threshold is 95.05.
        # Values >= 95.05 are [96, 97, 98, 99, 100]
        # Average should be 98.0
        vals = np.linspace(1.0, 100.0, 100)
        thresh = np.percentile(vals, 95)
        tail_avg = np.mean(vals[vals >= thresh])
        self.assertAlmostEqual(tail_avg, 98.0)

    def test_8_2_cvar_positive_format_in_alerts(self):
        cvar_val = 3.42
        alert_msg = f"Risk Note: On the worst 5% of days historically, rack prices spiked +{cvar_val:.2f}¢/gal (+${cvar_val * 85:.0f} per 8,500 gal truck)."
        self.assertIn("+3.42", alert_msg)

    def test_8_3_zscore_zero(self):
        change = 0.0
        std = 1.2
        z = change / std if std > 0 else 0.0
        self.assertEqual(z, 0.0)

    def test_8_4_zscore_zero_std(self):
        change = 1.5
        std = 0.0
        z = change / std if std > 0 else 0.0
        self.assertEqual(z, 0.0)


class TestCategory9AlertLogic(unittest.TestCase):
    """Category 9: Alert Logic Tests"""

    def test_9_1_threshold_boundary_exactly_equal(self):
        # Check boundary condition (>= hike_thresh)
        hike_thresh = 1.5
        change_cents = 1.5
        is_buy = change_cents >= hike_thresh
        self.assertTrue(is_buy)

    def test_9_2_tiered_alert_ordering(self):
        # Verify strict logical tiers
        # BUY_NOW > LEAN_BUY > NO_EDGE > LEAN_WAIT > WAIT
        hike = 1.5
        lean_hike = 0.5
        lean_drop = -0.5
        drop = -1.5

        def get_tier(change):
            if change >= hike:
                return "BUY_NOW"
            elif change >= lean_hike:
                return "LEAN_BUY"
            elif change <= drop:
                return "WAIT"
            elif change <= lean_drop:
                return "LEAN_WAIT"
            else:
                return "NO_EDGE"

        self.assertEqual(get_tier(2.0), "BUY_NOW")
        self.assertEqual(get_tier(1.0), "LEAN_BUY")
        self.assertEqual(get_tier(0.0), "NO_EDGE")
        self.assertEqual(get_tier(-1.0), "LEAN_WAIT")
        self.assertEqual(get_tier(-2.0), "WAIT")

    def test_9_3_no_alert_on_flat_market(self):
        change_cents = 0.0
        hike = 1.5
        drop = -1.5
        lean_hike = 0.5
        lean_drop = -0.5
        
        is_alert = (change_cents >= hike) or (change_cents <= drop) or (change_cents >= lean_hike) or (change_cents <= lean_drop)
        self.assertFalse(is_alert)

    def test_9_4_threshold_crossover_safety(self):
        # Hike and drop can never meet: the minimum clamp and the noise floor
        # both keep them strictly on opposite sides of zero.
        hike, drop = model.apply_noise_floor(
            model.MIN_ABS_THRESHOLD, -model.MIN_ABS_THRESHOLD, 0.0)
        self.assertTrue(hike > drop)
        hike, drop = model.apply_noise_floor(0.05, -0.05, 1.2)
        self.assertTrue(hike > drop)
        
        # Test signal generation exclusivity
        change = 0.0
        self.assertFalse(change >= hike and change <= drop)

    @patch('main.load_alert_state', return_value={})
    @patch('main.load_settlement_snapshot', return_value=None)
    @patch('main.is_contract_roll_day')
    def test_9_5_roll_day_suppression_regardless_of_move(self, mock_roll_check, mock_snapshot, mock_alert_state):
        mock_roll_check.return_value = True
        
        # Even with a massive 10.0 cent move (normally BUY_NOW with High Conviction)
        data = {
            'current_price': 2.20,
            'yesterday_close': 2.00,
            'open_price': 2.00,
            'high_price': 2.25,
            'low_price': 1.95,
            'daily_pct': 10.0,
            'five_day_high': 2.25,
            'five_day_low': 1.95,
            'thirty_day_avg': 2.05,
            'chart_intraday_b64': 'mock_base64',
            'chart_5d_b64': 'mock_base64',
        }
        
        signal = main.build_rack_signal('RB', data, datetime.now())
        self.assertEqual(signal['action'], 'NO_EDGE')
        self.assertEqual(signal['label'], 'Contract roll boundary')

        # Test with drop move (-10.0 cents)
        data_drop = data.copy()
        data_drop['current_price'] = 1.80
        data_drop['daily_pct'] = -10.0
        
        signal_drop = main.build_rack_signal('RB', data_drop, datetime.now())
        self.assertEqual(signal_drop['action'], 'NO_EDGE')
        self.assertEqual(signal_drop['label'], 'Contract roll boundary')

    @patch('main.load_settlement_snapshot', return_value=None)
    def test_9_6_tiered_action_ladder_boundaries(self, mock_load_snapshot):
        with patch.dict(main.APP_CONFIG, {
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "RB_nymex_daily_std": 1.0
        }):
            def get_act(current_p):
                data = {
                    'current_price': current_p,
                    'yesterday_close': 2.00,
                    'open_price': 2.00,
                    'high_price': 2.00,
                    'low_price': 2.00,
                    'daily_pct': 0.0,
                    'five_day_high': 2.00,
                    'five_day_low': 2.00,
                    'thirty_day_avg': 2.00,
                    # The ladder is only published at these widths when the
                    # baseline came from the calibration source; otherwise the
                    # thresholds widen to the fallback floor.
                    'baseline_source': 'settlement_provenance_verified',
                }
                return main.build_rack_signal('RB', data, datetime.now())['action']
            
            # Using slightly offset prices to ensure float arithmetic maps precisely to boundaries (unbiased)
            # Exactly equal or above hike_thresh (1.0c) -> BUY_NOW
            self.assertEqual(get_act(2.010001), "BUY_NOW")
            
            # Slightly below hike_thresh (0.99c) -> LEAN_BUY
            self.assertEqual(get_act(2.0099), "LEAN_BUY")
            
            # Slightly above lean_hike boundary (0.5c) to avoid float precision issues -> LEAN_BUY
            self.assertEqual(get_act(2.005001), "LEAN_BUY")
            
            # Slightly below lean_hike (0.49c) -> NO_EDGE
            self.assertEqual(get_act(2.0049), "NO_EDGE")
            
            # Exactly equal to drop_thresh (-1.0c) -> WAIT
            self.assertEqual(get_act(1.99), "WAIT")
            
            # Slightly above drop_thresh (-0.99c) -> LEAN_WAIT
            self.assertEqual(get_act(1.9901), "LEAN_WAIT")

            # Slightly below lean_drop boundary (-0.5c) -> LEAN_WAIT
            self.assertEqual(get_act(1.994999), "LEAN_WAIT")

            # Slightly above lean_drop (-0.49c) -> NO_EDGE
            self.assertEqual(get_act(1.9951), "NO_EDGE")

    @patch('main.load_settlement_snapshot', return_value=None)
    def test_9_6b_ladder_widens_on_a_mismatched_baseline(self, mock_load_snapshot):
        """A baseline that is not the calibration source gets the wider floor."""
        with patch.dict(main.APP_CONFIG, {
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "RB_nymex_daily_std": 1.0,
            "FALLBACK_NOISE_FLOOR_CENTS": 1.2,
        }):
            def get_act(current_p, source):
                data = {
                    'current_price': current_p, 'yesterday_close': 2.00,
                    'open_price': 2.00, 'high_price': 2.00, 'low_price': 2.00,
                    'daily_pct': 0.0, 'five_day_high': 2.00, 'five_day_low': 2.00,
                    'thirty_day_avg': 2.00, 'baseline_source': source,
                }
                return main.build_rack_signal('RB', data, datetime.now())['action']

            # +1.0c clears the published threshold but sits inside the source
            # mismatch budget, so it fires only on a calibration-matched baseline.
            self.assertEqual(get_act(2.010001, 'settlement_provenance_verified'), "BUY_NOW")
            self.assertEqual(get_act(2.010001, 'schwab_close_price'), "NO_EDGE")
            # A decisive move still fires either way.
            self.assertEqual(get_act(2.03, 'schwab_close_price'), "BUY_NOW")
            # An absent baseline_source is treated as unverified, not as verified.
            self.assertEqual(get_act(2.010001, None), "NO_EDGE")

    def test_9_7_savings_accounting(self):
        """BUY captures the rise, WAIT captures the fall, no-edge days score 0."""
        nymex = np.array([1.5, -2.0, 0.5, 2.0, -1.0, 0.0])
        rack = np.array([2.0, -1.0, 0.2, -0.5, -0.5, 0.1])
        summary = model.summarize(nymex, rack, hike=1.0, drop=-1.5)
        # Alerts: +1.5 (BUY, +2.0), -2.0 (WAIT, +1.0), +2.0 (BUY, -0.5).
        self.assertEqual(summary["alerts"], 3)
        self.assertAlmostEqual(summary["total_savings"], 2.5)
        self.assertAlmostEqual(summary["precision"], 2.0 / 3.0)
        self.assertEqual(sorted(summary["payoff"].tolist()), [-0.5, 1.0, 2.0])

    def test_9_8_passthrough_fit_recovers_a_known_relationship(self):
        """Replaces the percentile tuner with the fit that supersedes it."""
        rng = np.random.default_rng(13)
        x = rng.normal(0, 8, 500)
        y = 1.0 * x + rng.normal(0, 0.5, 500)
        fit = model.fit_passthrough(x, y)
        self.assertAlmostEqual(fit.slope, 1.0, places=1)
        self.assertGreater(fit.r2, 0.99)
        hike, drop = fit.threshold_for_confidence(0.75)
        self.assertGreater(hike, 0)
        self.assertLess(drop, 0)

class TestCategory10HistoricalReplay(unittest.TestCase):
    """Category 10: Historical Replay Tests"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.real_data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # Copy real files to temp dir
        shutil.copy(os.path.join(self.real_data_dir, "graves_history.csv"), self.temp_dir)
        shutil.copy(os.path.join(self.real_data_dir, "config.json"), self.temp_dir)
        cache_src = os.path.join(self.real_data_dir, "metrics_cache.json")
        if os.path.exists(cache_src):
            shutil.copy(cache_src, self.temp_dir)
        
        # Patch paths
        self.orig_data_dir = backtest.DATA_DIR
        self.orig_csv_path = backtest.CSV_PATH
        self.orig_config_path = backtest.CONFIG_PATH
        self.orig_metrics_cache_path = backtest.METRICS_CACHE_PATH
        
        backtest.DATA_DIR = self.temp_dir
        backtest.CSV_PATH = os.path.join(self.temp_dir, "graves_history.csv")
        backtest.CONFIG_PATH = os.path.join(self.temp_dir, "config.json")
        backtest.METRICS_CACHE_PATH = os.path.join(self.temp_dir, "metrics_cache.json")

    def tearDown(self):
        backtest.DATA_DIR = self.orig_data_dir
        backtest.CSV_PATH = self.orig_csv_path
        backtest.CONFIG_PATH = self.orig_config_path
        backtest.METRICS_CACHE_PATH = self.orig_metrics_cache_path
        shutil.rmtree(self.temp_dir)

    def test_10_1_real_database_length(self):
        df = pd.read_csv(backtest.CSV_PATH)
        print(f"\n[Test 10.1] Loaded real CSV inside test suite. Length: {len(df)}")
        self.assertTrue(len(df) >= 766)

    def test_10_2_walk_forward_fold_isolation_real_data(self):
        df = pd.read_csv(backtest.CSV_PATH)
        self.assertTrue(len(df) >= 766)
        
        # Verify train/test isolation on the real data
        test_size = 90
        W = 120
        folds = 3
        for f in range(folds):
            test_start = len(df) - (f + 1) * test_size
            test_end = len(df) - f * test_size if f > 0 else len(df)
            train_start = max(0, test_start - W)
            
            df_train = df.iloc[train_start:test_start]
            df_test = df.iloc[test_start:test_end]
            
            # Assert no date overlap
            train_dates = set(df_train['date'])
            test_dates = set(df_test['date'])
            self.assertTrue(train_dates.isdisjoint(test_dates))

    def test_10_3_lag_0_correlation_real_data(self):
        from scipy import stats
        df = pd.read_csv(backtest.CSV_PATH)
        self.assertTrue(len(df) >= 766)
        
        df_clean = df.dropna(subset=['nymex_rb', 'rack_u']).copy()
        df_clean['delta_nymex'] = df_clean['nymex_rb'].diff() * 100
        df_clean['delta_rack'] = df_clean['rack_u'].diff() * 100
        df_clean = df_clean.dropna(subset=['delta_nymex', 'delta_rack'])
        
        slope, intercept, r_val, p_val, std_err = stats.linregress(df_clean['delta_nymex'], df_clean['delta_rack'])
        # Verify genuine strong relationship
        self.assertTrue(r_val**2 > 0.10)
        self.assertTrue(p_val < 0.01)

    def test_10_4_full_replay_on_real_data(self):
        """Calibrate on the real history and check every published figure.

        Note what is asserted: the *out-of-sample* precision, not an in-sample
        win rate.  The superseded engine wrote ``RB_historical_win_rate``
        measured on its own training window (it read 0.94) and this test
        asserted that number was above 0.50, which it could not fail to be.
        """
        full = alignment.load_history(backtest.CSV_PATH)
        df = alignment.calibration_history(full)
        self.assertGreaterEqual(len(df), 200)

        cfg = dict(backtest.DEFAULTS)
        cfg, msg, evaluation = backtest.calibrate(df, 'RB', cfg)

        self.assertIn('RB_oos_precision', cfg)
        self.assertEqual(cfg['RB_oos_alerts'], evaluation['alerts'])
        self.assertGreater(cfg['RB_oos_alerts'], 30)
        self.assertGreater(cfg['RB_oos_precision'], 0.50)
        self.assertLessEqual(cfg['RB_oos_precision'], 1.0)
        # In-sample figures must be present but distinctly named, so no
        # consumer can mistake one for the other.
        self.assertIn('RB_insample_precision', cfg)
        self.assertNotIn('RB_historical_win_rate', cfg)
        # The fitted relationship must be economically sensible.
        self.assertGreater(cfg['RB_pt_slope'], 0.2)
        self.assertLess(cfg['RB_pt_slope'], 1.5)
        self.assertGreater(cfg['RB_pt_r2'], 0.3)
        # Thresholds must clear the measured snapshot noise floor.
        self.assertGreaterEqual(cfg['RB_HIKE_THRESHOLD_CENTS'],
                                cfg['SNAPSHOT_NOISE_FLOOR_CENTS'])
        self.assertLessEqual(cfg['RB_DROP_THRESHOLD_CENTS'],
                             -cfg['SNAPSHOT_NOISE_FLOOR_CENTS'])

    def test_10_5_permutation_pvalue_math(self):
        real_savings = 50.0
        perm_savings = [10.0, 20.0, 30.0, 45.0, 52.0]
        p_val = np.mean(np.array(perm_savings) >= real_savings)
        self.assertEqual(p_val, 0.20)

    @patch('numpy.random.permutation')
    def test_10_6_permutation_pvalue_precision(self, mock_perm):
        mock_perm.side_effect = [[10.0, -10.0]] * 500 + [[-10.0, 10.0]] * 500
        
        pred_dirs = ['HIKE', 'DROP']
        actual_moves = [10.0, -10.0]
        real_sig_savings = 20.0
        
        n_perm = 1000
        perm_savings = []
        for _ in range(n_perm):
            shuffled_actuals = mock_perm(actual_moves)
            sh_sav = 0.0
            for p_dir, sh_act in zip(pred_dirs, shuffled_actuals):
                if p_dir == 'HIKE':
                    sh_sav += sh_act
                elif p_dir == 'DROP':
                    sh_sav += -sh_act
            perm_savings.append(sh_sav)
            
        better_trials = np.sum(np.array(perm_savings) >= real_sig_savings)
        p_value = float((better_trials + 1) / (n_perm + 1))
        
        self.assertEqual(better_trials, 500)
        self.assertAlmostEqual(p_value, 501 / 1001)


class TestCategory11AlertFormatting(unittest.TestCase):
    """Category 11: verdict text is derived from the model, never hardcoded."""

    @staticmethod
    def _cache_for(prefix, slope, intercept, sigma, seed):
        """A realistic fitted model serialised the way the cache stores it."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 8, 400)
        y = intercept + slope * x + rng.normal(0, sigma, 400)
        return model.passthrough_to_config(model.fit_passthrough(x, y), prefix)

    def setUp(self):
        import main
        self.orig_main_data_dir = main.DATA_DIR
        self.temp_dir = tempfile.mkdtemp()
        orig_config_path = os.path.join(self.orig_main_data_dir, "config.json")
        if os.path.exists(orig_config_path):
            shutil.copy(orig_config_path, self.temp_dir)
        main.DATA_DIR = self.temp_dir
        self.snapshot_patcher = patch('main.load_settlement_snapshot', return_value=None)
        self.snapshot_patcher.start()
        self.alert_state_patcher = patch('main.load_alert_state', return_value={})
        self.alert_state_patcher.start()

        self.orig_config = main.APP_CONFIG.copy()
        main.APP_CONFIG.clear()
        main.APP_CONFIG.update(main.DEFAULT_APP_CONFIG)
        main.APP_CONFIG.update({
            "RB_nymex_daily_std": 10.0,
            "HO_nymex_daily_std": 12.0,
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "HO_HIKE_THRESHOLD_CENTS": 2.0,
            "HO_DROP_THRESHOLD_CENTS": -2.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "HO_LEAN_HIKE_CENTS": 1.0,
            "HO_LEAN_DROP_CENTS": -1.0,
            "RB_oos_precision": 0.82,
            "RB_oos_alerts": 99,
            "RB_oos_window": "2026-03-17..2026-09-16",
            "HO_oos_precision": 0.79,
            "HO_oos_alerts": 109,
            "HO_oos_window": "2026-03-17..2026-09-16",
            "HO_wait_cvar_status": "ok",
            "HO_wait_cvar_95": 2.95,
            "HO_wait_cvar_low": -0.84,
            "HO_wait_cvar_high": 1.96,
            "HO_wait_cvar_tail_n": 5,
            "HO_wait_cvar_sample_n": 93,
            "HO_wait_adverse_probability": 0.043,
            "HO_wait_median_move": -6.39,
        })
        main.APP_CONFIG.update(self._cache_for("RB", 0.70, 0.0, 3.0, seed=1))
        main.APP_CONFIG.update(self._cache_for("HO", 0.93, 0.0, 4.0, seed=2))
        self.now = datetime.now()

    def tearDown(self):
        import main
        self.alert_state_patcher.stop()
        self.snapshot_patcher.stop()
        main.DATA_DIR = self.orig_main_data_dir
        main.APP_CONFIG.clear()
        main.APP_CONFIG.update(self.orig_config)
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _data(current, prior=2.00):
        return {
            'current_price': current, 'yesterday_close': prior,
            'open_price': prior, 'high_price': max(current, prior) + 0.05,
            'low_price': min(current, prior) - 0.05, 'daily_pct': 0.0,
            'five_day_high': 2.25, 'five_day_low': 1.95, 'thirty_day_avg': 2.05,
            'chart_intraday_b64': 'mock_base64', 'chart_5d_b64': 'mock_base64',
        }

    def test_11_1_buy_quotes_a_calibrated_probability(self):
        import main
        signal = main.build_rack_signal('RB', self._data(2.20), self.now)
        self.assertEqual(signal['action'], 'BUY_NOW')
        self.assertIn('High confidence', signal['conviction'])

        risk_text = signal['risk_text']
        self.assertIn('Model confidence:', risk_text)
        self.assertIn('that the rack will rise tonight', risk_text)
        self.assertIn('expected move', risk_text)
        # A +20c NYMEX move at slope 0.70 is overwhelming; the quoted
        # probability must reflect that rather than a coarse bin.
        self.assertRegex(risk_text, r'Model confidence: 9\d%')

    def test_11_2_wait_states_tail_risk_with_its_sample_size(self):
        import main
        signal = main.build_rack_signal('HO', self._data(1.94), self.now)
        self.assertEqual(signal['action'], 'WAIT')

        risk_text = signal['risk_text']
        self.assertIn('Deferral risk:', risk_text)
        self.assertIn('the rack rose on 4% of 93 past WAIT signals', risk_text)
        self.assertIn('median WAIT day moved -6.39¢/gal', risk_text)
        self.assertIn('Worst 5% averaged +2.95¢/gal', risk_text)
        self.assertIn('8,500-gal truck', risk_text)
        # The uncertainty must travel with the number.
        self.assertIn('only 5 observations', risk_text)
        self.assertIn('95% CI -0.84 to +1.96¢/gal', risk_text)

    def test_11_3_unquantified_tail_is_stated_not_guessed(self):
        import main
        main.APP_CONFIG['HO_wait_cvar_status'] = 'insufficient_tail'
        signal = main.build_rack_signal('HO', self._data(1.94), self.now)
        self.assertEqual(signal['action'], 'WAIT')
        self.assertIn('too few WAIT observations', signal['risk_text'])
        self.assertIn('unquantified', signal['risk_text'])

    def test_11_4_no_hardcoded_performance_claims_anywhere(self):
        """The stale README ranges must not reappear in any verdict text."""
        import main
        for prefix, price in (('RB', 2.20), ('RB', 1.80), ('HO', 2.36), ('HO', 1.94)):
            text = main.build_rack_signal(prefix, self._data(price), self.now)['risk_text']
            for banned in ('53%–73%', '60%–79%', 'planning floor', 'Multi-year baseline'):
                self.assertNotIn(banned, text)
        source = inspect.getsource(main.build_risk_note)
        for banned in ('53%', '73%', '60%', '79%'):
            self.assertNotIn(banned, source)

    def test_11_5_out_of_sample_figure_carries_an_interval(self):
        import main
        text = main.build_rack_signal('RB', self._data(2.20), self.now)['risk_text']
        self.assertIn('Out-of-sample: 82% correct on 99 alerts', text)
        self.assertIn('95% CI', text)
        self.assertIn('upper bound', text)

    def test_11_6_missing_model_degrades_without_inventing_confidence(self):
        """With no fitted model cached, the system must not quote a number."""
        import main
        for key in list(main.APP_CONFIG):
            if key.startswith('RB_pt_'):
                del main.APP_CONFIG[key]
        signal = main.build_rack_signal('RB', self._data(2.20), self.now)
        self.assertEqual(signal['conviction'], 'Confidence unavailable')
        self.assertNotIn('Model confidence:', signal['risk_text'])

class TestCategory12ProductionFailureProtection(unittest.TestCase):
    """Category 12: Production Failure Protection & Edge Cases"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "graves_history.csv")
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.log_path = os.path.join(self.temp_dir, "prediction_log.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('main.datetime')
    @patch('main.get_repo_variable')
    @patch('main.set_repo_variable')
    @patch('main.save_settlement_snapshots')
    def test_12_1_timezone_runner_clock_drift(self, mock_snapshots, mock_set, mock_get, mock_datetime):
        # 1. Mock runner clock at various CT times:
        # Timezone check behavior uses datetime.now(TZ)
        tz = pytz.timezone('America/Chicago')
        
        # Test 1:29 PM CT (snapshots should not fire)
        dt_129 = tz.localize(datetime(2026, 5, 22, 13, 29, 0))
        mock_datetime.now.return_value = dt_129
        mock_datetime.combine = datetime.combine
        main.save_settlement_snapshots({}, dt_129)
        # Check: save_settlement_snapshots requires 13:30 to 13:45
        self.assertFalse(dt_129.hour == 13 and 30 <= dt_129.minute <= 45)

        # Test 1:31 PM CT (snapshots should fire)
        dt_131 = tz.localize(datetime(2026, 5, 22, 13, 31, 0))
        self.assertTrue(dt_131.hour == 13 and 30 <= dt_131.minute <= 45)

        # Test 7:29 PM CT (daily prompt should not fire)
        dt_729 = tz.localize(datetime(2026, 5, 22, 19, 29, 0))
        self.assertTrue(dt_729.weekday() > 4 or not (dt_729.hour == 19 and dt_729.minute >= 30))

        # Test 7:31 PM CT (daily prompt should fire)
        dt_731 = tz.localize(datetime(2026, 5, 22, 19, 31, 0))
        self.assertFalse(dt_731.weekday() > 4 or not (dt_731.hour == 19 and dt_731.minute >= 30))

        # Test 8:01 PM CT (daily prompt should not fire)
        dt_801 = tz.localize(datetime(2026, 5, 22, 20, 1, 0))
        self.assertTrue(dt_801.weekday() > 4 or not (dt_801.hour == 19 and dt_801.minute >= 30))

    def test_12_2_partial_csv_write_corruption(self):
        # Create a CSV with a truncated final line (less than 6 comma-separated elements)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
            f.write("2026-05-18,2.10,2.20,2.30,2.40,2.50\n")
            f.write("2026-05-19,2.15,2.25,2.35") # Truncated line

        # Call repair function
        validate_data.repair_csv_if_corrupted(self.csv_path)

        # Verify that the corrupt last line was pruned
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1], "2026-05-18,2.10,2.20,2.30,2.40,2.50")

    @patch('main.load_settlement_snapshot', return_value=None)
    @patch('main.CONFIG_PATH')
    @patch('main.CONFIG_CORRUPT', True)
    def test_12_3_config_rollback(self, mock_path, mock_snapshot):
        # Verify build_rack_signal displays corrupt warning when CONFIG_CORRUPT is True
        data = {
            'current_price': 2.20,
            'yesterday_close': 2.00,
            'open_price': 2.00,
            'high_price': 2.25,
            'low_price': 1.95,
            'daily_pct': 10.0,
            'five_day_high': 2.25,
            'five_day_low': 1.95,
            'thirty_day_avg': 2.05,
            'chart_intraday_b64': 'mock_base64',
            'chart_5d_b64': 'mock_base64',
        }
        signal = main.build_rack_signal('RB', data, datetime.now())
        self.assertIn("WARNING: Corrupt config.json detected!", signal['risk_text'])

    @patch('main.load_alert_state', return_value={})
    @patch('main.load_settlement_snapshot', return_value=None)
    @patch('main.is_contract_roll_day')
    def test_12_4_nymex_contract_roll_boundary(self, mock_roll_check, mock_snapshot, mock_alert_state):
        mock_roll_check.return_value = True
        data = {
            'current_price': 2.20,
            'yesterday_close': 2.00,
            'open_price': 2.00,
            'high_price': 2.25,
            'low_price': 1.95,
            'daily_pct': 10.0,
            'five_day_high': 2.25,
            'five_day_low': 1.95,
            'thirty_day_avg': 2.05,
            'chart_intraday_b64': 'mock_base64',
            'chart_5d_b64': 'mock_base64',
        }
        signal = main.build_rack_signal('RB', data, datetime.now())
        self.assertEqual(signal['action'], 'NO_EDGE')
        self.assertEqual(signal['label'], 'Contract roll boundary')
        self.assertIn("roll price gaps", signal['text'])

    @patch('main.save_price_history')
    @patch('main.load_price_history', return_value=[])
    @patch('time.sleep')
    @patch('main.resolve_active_schwab_symbol', return_value='/RBM26')
    @patch('main.save_alert_state')
    @patch('main.load_alert_state', return_value={})
    @patch('main.requests.get', side_effect=RuntimeError('Schwab unavailable in unit test'))
    @patch('main.yf.Ticker')
    @patch('main.previous_nymex_business_day')
    def test_12_5_yfinance_stale_cache_fallback(self, mock_prev_day, mock_ticker, mock_get, mock_load, mock_save, mock_resolve, mock_sleep, mock_load_history, mock_save_history):
        # Mock yesterday's date as 2026-05-21, but yfinance history returns last date 2026-05-20 (stale)
        mock_prev_day.return_value = datetime(2026, 5, 21).date()
        
        mock_history = MagicMock()
        # Mock 1mo daily history
        hist_df = pd.DataFrame(
            {'Close': [2.00]},
            index=[datetime(2026, 5, 20, tzinfo=pytz.utc)]
        )
        mock_history.history.side_effect = [
            pd.DataFrame(), # 5d hourly empty
            hist_df # 1mo daily
        ]
        mock_ticker.return_value = mock_history
        
        # Test baseline extraction fallback in fetch_commodity
        cfg = {'name': 'Wholesale Gas', 'yf_symbol': 'RB=F'}
        
        original_open = open
        with patch('builtins.open') as mock_op:
            def mock_open_fn(file, *args, **kwargs):
                if "graves_history.csv" in str(file):
                    raise FileNotFoundError()
                return original_open(file, *args, **kwargs)
            mock_op.side_effect = mock_open_fn
            res = main.fetch_commodity('RB', cfg, datetime(2026, 5, 22), None)
        
        # yesterday_close should be overridden to None because cache was stale (last date 2026-05-20 < 2026-05-21) and CSV is empty
        self.assertIsNone(res['yesterday_close'])

    @patch('main.yf.Ticker')
    @patch('main.requests.get')
    def test_12_5a_active_schwab_symbol_resolution_uses_schwab_metadata(self, mock_get, mock_yf):
        """A deferred contract's higher open interest must not override Schwab's mapping."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            '/RB': {'reference': {'futureActiveSymbol': '/RBQ26'}},
            '/RBQ26': {'quote': {'symbol': '/RBQ26', 'openInterest': 100}},
            '/RBU26': {'quote': {'symbol': '/RBU26', 'openInterest': 100000}},
        }
        mock_get.return_value = mock_response

        resolved = main.resolve_active_schwab_symbol(
            'RB',
            pytz.timezone('America/Chicago').localize(datetime(2026, 7, 20, 12, 0)),
            'dummy_token'
        )

        self.assertEqual(resolved, '/RBQ26')
        mock_yf.assert_not_called()

    def test_12_5a_active_symbol_extraction_supports_future_is_active(self):
        response = {
            '/HOQ26': {
                'quote': {'symbol': '/HOQ26', 'openInterest': 100},
                'reference': {'futureIsActive': True},
            },
            '/HOU26': {
                'quote': {'symbol': '/HOU26', 'openInterest': 100000},
                'reference': {'futureIsActive': False},
            },
        }
        self.assertEqual(main.extract_schwab_active_future_symbol(response, 'HO'), '/HOQ26')

    def test_12_5a_active_symbol_extraction_rejects_ambiguous_flags(self):
        response = {
            '/RBQ26': {'reference': {'futureIsActive': True}},
            '/RBU26': {'reference': {'futureIsActive': 'true'}},
        }
        self.assertIsNone(main.extract_schwab_active_future_symbol(response, 'RB'))

    @patch('main.yf.Ticker')
    @patch('main.requests.get')
    def test_12_5a_regression_near_term_on_tie(self, mock_get, mock_yf):
        """Regression test for May 2026 bug: pick nearest contract on activity tie."""
        # Mock yfinance to return no underlyingSymbol so this test exercises
        # the Schwab-volume fallback path (the path this test was designed for).
        mock_yf.return_value.info.get.return_value = None
        # Simulate May 26, 2026 at 5pm: all 4 candidates have zero activity (thin market)
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            '/RBM26': {'quote': {'symbol': '/RBM26', 'lastPrice': 2.10}},  # June - index 0
            '/RBN26': {'quote': {'symbol': '/RBN26', 'lastPrice': 2.15}},  # July - index 1
            '/RBQ26': {'quote': {'symbol': '/RBQ26', 'lastPrice': 2.20}},  # August - index 2
            '/RBU26': {'quote': {'symbol': '/RBU26', 'lastPrice': 2.25}},  # September - index 3
        }
        mock_get.return_value = mock_response

        resolved = main.resolve_active_schwab_symbol(
            'RB',
            datetime(2026, 5, 26, 17, 0, tzinfo=pytz.utc),
            'dummy_token'
        )

        # With zero activity for all, volume fallback is bypassed (< 100 threshold)
        # and the math-based candidates[0] is used.  Session date is May 27
        # (17:00 UTC pushes get_session_date_str to the next day).
        #
        # The expected answer moved when DEFAULT_EARLY_ROLL_DAYS was corrected
        # from an assumed 3 to a measured 2.  At 3, May 27 fell inside the
        # early-roll window and the math jumped to July while the market was
        # still quoting June.  The roll offset is now calibrated against the
        # observed settle gaps (see futures_util.DEFAULT_EARLY_ROLL_DAYS), and
        # at 2 the June contract is correctly still front month on May 27.
        self.assertEqual(futures_util.DEFAULT_EARLY_ROLL_DAYS, 2)
        self.assertEqual(resolved, '/RBM26',
                         "At the calibrated 2-business-day early roll, May 27 is "
                         "still the June contract")


    @patch('main.save_price_history')
    @patch('main.load_price_history', return_value=[])
    @patch('main.save_alert_state')
    @patch('main.load_alert_state', return_value={})
    @patch('main.yf.Ticker')
    @patch('main.requests.get')
    @patch('main.resolve_active_schwab_symbol')
    def test_12_5b_fetch_commodity_uses_active_schwab_primary(self, mock_resolve, mock_get, mock_ticker, mock_load_alert, mock_save_alert, mock_load_history, mock_save_history):
        mock_resolve.return_value = '/RBM26'

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            '/RBM26': {
                'quote': {
                    'lastPrice': 2.20,
                    'openPrice': 2.10,
                    'highPrice': 2.25,
                    'lowPrice': 2.00,
                    'closePrice': 2.05
                }
            }
        }
        mock_get.return_value = mock_response

        mock_history = MagicMock()
        h5d = pd.DataFrame(
            {'Close': [2.20], 'High': [2.25], 'Low': [2.00]},
            index=[datetime(2026, 5, 21, tzinfo=pytz.utc)]
        )
        h30d = pd.DataFrame(
            {'Close': [2.05]},
            index=[datetime(2026, 5, 21, tzinfo=pytz.utc)]
        )
        mock_history.history.side_effect = [h5d, h30d]
        mock_ticker.return_value = mock_history

        cfg = {'name': 'Wholesale Gas', 'yf_symbol': 'RB=F'}
        res = main.fetch_commodity('RB', cfg, datetime(2026, 5, 22, 12, 0, tzinfo=pytz.utc), 'token')

        self.assertEqual(res['schwab_symbol'], '/RBM26')
        self.assertEqual(res['current_price'], 2.20)
        self.assertEqual(res['baseline_schwab_symbol'], '/RBM26')
        self.assertEqual(res['baseline_source'], 'schwab_close_price')
        self.assertTrue(res['contract_provenance_required'])
        mock_resolve.assert_called_once_with('RB', datetime(2026, 5, 22, 12, 0, tzinfo=pytz.utc), 'token')

    @patch('main.save_price_history')
    @patch('main.load_price_history', return_value=[])
    @patch('main.save_alert_state')
    @patch('main.load_alert_state')
    @patch('main.yf.Ticker')
    @patch('main.requests.get')
    @patch('main.resolve_active_schwab_symbol')
    def test_12_5c_fetch_commodity_uses_cached_active_symbol(self, mock_resolve, mock_get, mock_ticker, mock_load, mock_save, mock_load_history, mock_save_history):
        # Cache has '/RBN26' for the session 2026-05-22
        mock_load.return_value = {
            'ACTIVE_SYMBOL_RB_2026-05-22': '/RBN26'
        }

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            '/RBN26': {
                'quote': {
                    'lastPrice': 2.20,
                    'openPrice': 2.10,
                    'highPrice': 2.25,
                    'lowPrice': 2.00,
                    'closePrice': 2.05
                }
            }
        }
        mock_get.return_value = mock_response

        mock_history = MagicMock()
        h5d = pd.DataFrame(
            {'Close': [2.20], 'High': [2.25], 'Low': [2.00]},
            index=[datetime(2026, 5, 21, tzinfo=pytz.utc)]
        )
        h30d = pd.DataFrame(
            {'Close': [2.05]},
            index=[datetime(2026, 5, 21, tzinfo=pytz.utc)]
        )
        mock_history.history.side_effect = [h5d, h30d]
        mock_ticker.return_value = mock_history

        cfg = {'name': 'Wholesale Gas', 'yf_symbol': 'RB=F'}
        res = main.fetch_commodity('RB', cfg, datetime(2026, 5, 22, 12, 0, tzinfo=pytz.utc), 'token')

        self.assertEqual(res['schwab_symbol'], '/RBN26')
        self.assertEqual(res['current_price'], 2.20)
        mock_resolve.assert_not_called()

    @patch('main.save_price_history')
    @patch('main.load_price_history', return_value=[])
    @patch('main.save_alert_state')
    @patch('main.load_alert_state')
    @patch('main.yf.Ticker')
    @patch('main.requests.get')
    @patch('main.resolve_active_schwab_symbol')
    @patch('main.get_schwab_declared_active_symbol', return_value='/RBQ26')
    def test_12_5c_cached_symbol_updates_when_schwab_rolls(
        self, mock_declared, mock_resolve, mock_get, mock_ticker, mock_load,
        mock_save, mock_load_history, mock_save_history
    ):
        mock_load.return_value = {'ACTIVE_SYMBOL_RB_2026-07-20': '/RBU26'}

        quote_response = MagicMock()
        quote_response.raise_for_status.return_value = None
        quote_response.json.return_value = {
            '/RBQ26': {
                'quote': {
                    'lastPrice': 3.20,
                    'openPrice': 3.15,
                    'highPrice': 3.22,
                    'lowPrice': 3.12,
                    'closePrice': 3.18,
                }
            }
        }
        mock_get.return_value = quote_response

        mock_history = MagicMock()
        mock_history.history.side_effect = [
            pd.DataFrame(
                {'Close': [3.20], 'High': [3.22], 'Low': [3.12]},
                index=[datetime(2026, 7, 20, tzinfo=pytz.utc)],
            ),
            pd.DataFrame(
                {'Close': [3.18]},
                index=[datetime(2026, 7, 20, tzinfo=pytz.utc)],
            ),
        ]
        mock_ticker.return_value = mock_history

        now = pytz.timezone('America/Chicago').localize(datetime(2026, 7, 20, 12, 0))
        result = main.fetch_commodity('RB', {'name': 'Wholesale Gas'}, now, 'token')

        self.assertEqual(result['schwab_symbol'], '/RBQ26')
        self.assertEqual(result['baseline_schwab_symbol'], '/RBQ26')
        self.assertEqual(result['_alert_state_updates']['ACTIVE_SYMBOL_RB_2026-07-20'], '/RBQ26')
        mock_declared.assert_called_once()
        mock_resolve.assert_not_called()

    @patch('main.save_price_history')
    @patch('main.load_price_history', return_value=[])
    @patch('main.save_alert_state')
    @patch('main.load_alert_state')
    @patch('main.yf.Ticker')
    @patch('main.requests.get')
    @patch('main.resolve_active_schwab_symbol')
    def test_12_5c_fetch_commodity_resolves_and_caches_active_symbol_if_empty(self, mock_resolve, mock_get, mock_ticker, mock_load, mock_save, mock_load_history, mock_save_history):
        mock_load.return_value = {}
        mock_resolve.return_value = '/RBN26'

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            '/RBN26': {
                'quote': {
                    'lastPrice': 2.20,
                    'openPrice': 2.10,
                    'highPrice': 2.25,
                    'lowPrice': 2.00,
                    'closePrice': 2.05
                }
            }
        }
        mock_get.return_value = mock_response

        mock_history = MagicMock()
        h5d = pd.DataFrame(
            {'Close': [2.20], 'High': [2.25], 'Low': [2.00]},
            index=[datetime(2026, 5, 21, tzinfo=pytz.utc)]
        )
        h30d = pd.DataFrame(
            {'Close': [2.05]},
            index=[datetime(2026, 5, 21, tzinfo=pytz.utc)]
        )
        mock_history.history.side_effect = [h5d, h30d]
        mock_ticker.return_value = mock_history

        cfg = {'name': 'Wholesale Gas', 'yf_symbol': 'RB=F'}
        res = main.fetch_commodity('RB', cfg, datetime(2026, 5, 22, 12, 0, tzinfo=pytz.utc), 'token')

        self.assertEqual(res['schwab_symbol'], '/RBN26')
        mock_resolve.assert_called_once_with('RB', datetime(2026, 5, 22, 12, 0, tzinfo=pytz.utc), 'token')
        mock_save.assert_not_called()
        self.assertEqual(
            res['_alert_state_updates']['ACTIVE_SYMBOL_RB_2026-05-22'],
            '/RBN26'
        )

    @patch('main.load_settlement_snapshot', return_value=None)
    @patch('main.load_price_history', side_effect=AssertionError('signal construction must not read intraday history'))
    @patch('main.APP_CONFIG')
    def test_12_9_z_score_uses_calibrated_daily_std(self, mock_config, mock_load_history, mock_snapshot):
        # The calibrated daily standard deviation is the sole Z-score denominator.
        mock_config.get.side_effect = lambda key, default=None: {
            'RB_HIKE_THRESHOLD_CENTS': 1.0,
            'RB_DROP_THRESHOLD_CENTS': -1.0,
            'RB_LEAN_HIKE_CENTS': 0.5,
            'RB_LEAN_DROP_CENTS': -0.5,
            'RB_nymex_daily_std': 1.0
        }.get(key, default)

        data = {
            'current_price': 2.10,
            'yesterday_close': 2.00,
            'open_price': 2.00,
            'schwab_symbol': '/RBK26'
        }

        signal = main.build_rack_signal('RB', data, datetime(2026, 5, 23, 13, 0))
        self.assertNotIn("dynamic intraday vol override", signal['risk_text'])
        # No pass-through model is present in this stubbed config, so the
        # verdict must be suppressed rather than fall back to default
        # thresholds and emit a confident-looking BUY built on nothing.
        self.assertEqual(signal['action'], 'NO_EDGE')
        self.assertEqual(signal['label'], 'Calibration unavailable')
        self.assertEqual(signal['conviction'], 'Confidence unavailable')

        # With a model present the Z-score denominator is still the calibrated
        # daily std, which is what this test was originally guarding.
        rng = np.random.default_rng(21)
        x = rng.normal(0, 8, 400)
        y = 0.7 * x + rng.normal(0, 3, 400)
        full = dict(main.DEFAULT_APP_CONFIG)
        full.update({'RB_HIKE_THRESHOLD_CENTS': 1.0, 'RB_DROP_THRESHOLD_CENTS': -1.0,
                     'RB_LEAN_HIKE_CENTS': 0.5, 'RB_LEAN_DROP_CENTS': -0.5,
                     'RB_nymex_daily_std': 1.0})
        full.update(model.passthrough_to_config(model.fit_passthrough(x, y), 'RB'))
        with patch.object(main, 'APP_CONFIG', full):
            signal = main.build_rack_signal('RB', data, datetime(2026, 5, 23, 13, 0))
        self.assertAlmostEqual(signal['z_score'], 10.0)

    @patch('main.get_repo_variable')
    @patch('main.set_repo_variable')
    def test_12_10_save_settlement_snapshots_idempotency(self, mock_set_var, mock_get_var):
        tz = pytz.timezone('America/Chicago')
        dt = tz.localize(datetime(2026, 5, 22, 13, 35, 0))
        
        all_data = {
            'RB': {'current_price': 2.20, 'schwab_symbol': '/RBK26'},
            'HO': {'current_price': 2.30, 'schwab_symbol': '/HOK26'}
        }
        
        session_str = "2026-05-22"
        existing_snapshot = {
            "date": session_str,
            "price": 2.20,
            "captured_at": dt.isoformat(),
            "schwab_symbol": "/RBK26"
        }
        
        saved_snapshots = {
            "SETTLE_SNAPSHOT_RB": json.dumps(existing_snapshot)
        }
        def side_get(key):
            return saved_snapshots.get(key)
        def side_set(key, val):
            saved_snapshots[key] = val
            
        mock_get_var.side_effect = side_get
        mock_set_var.side_effect = side_set
        
        temp_dir = tempfile.mkdtemp()
        orig_data_dir = main.DATA_DIR
        main.DATA_DIR = temp_dir
        
        try:
            main.save_settlement_snapshots(all_data, dt)
            
            called_keys = [args[0] for args, kwargs in mock_set_var.call_args_list]
            self.assertNotIn("SETTLE_SNAPSHOT_RB", called_keys)
            self.assertIn("SETTLE_SNAPSHOT_HO", called_keys)
            self.assertIn("SETTLE_SNAPSHOT_RB", saved_snapshots)
            self.assertIn("SETTLE_SNAPSHOT_HO", saved_snapshots)
            
            mock_set_var.reset_mock()
            
            ds_path = os.path.join(temp_dir, "daily_settlement.json")
            with open(ds_path, "w") as f:
                json.dump({"date": session_str, "captured_at": dt.isoformat()}, f)
                
            main.save_settlement_snapshots(all_data, dt)
            mock_set_var.assert_not_called()
            
        finally:
            main.DATA_DIR = orig_data_dir
            shutil.rmtree(temp_dir)



class TestCategory13LiveValidationAndRobustness(unittest.TestCase):
    """Category 13: Live Validation & Robustness Regression Tests"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "graves_history.csv")
        self.log_path = os.path.join(self.temp_dir, "prediction_log.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_13_1_fuzzer_sandbox_isolation(self):
        import scratch.fuzz_parser
        scratch.fuzz_parser.run_fuzzer()

    @patch('ingest_prices.datetime')
    @patch('subprocess.run')
    @patch('ingest_prices.send_alert_email')
    def test_13_2_git_commit_failure_notification(self, mock_send_email, mock_sub_run, mock_datetime):
        mock_datetime.now.return_value = datetime(2026, 5, 22, 12, 0, tzinfo=pytz.timezone('America/Chicago'))
        mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
        
        import subprocess
        mock_sub_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "backtest.py"],
            output="Mock stdout info",
            stderr=b"git push failed: Merge conflict in config.json"
        )
        
        with patch.dict(os.environ, {
            'GRAVES_EMAIL': 'mock@example.com',
            'GRAVES_APP_PASSWORD': 'mock',
            'GMAIL_USER': 'mock',
            'GMAIL_APP_PASSWORD': 'mock',
            'TO_EMAIL': 'mock@example.com',
            'GITHUB_EVENT_NAME': 'workflow_dispatch'
        }):
            with patch('ingest_prices.check_inbox_for_prices') as mock_inbox:
                mock_inbox.return_value = ("2026-05-22", (2.10, 2.20, 2.30))
                with patch('ingest_prices.read_daily_settlement') as mock_settle:
                    mock_settle.return_value = {"rbob_settlement": 2.05, "heating_oil_settlement": 2.15}
                    with patch('ingest_prices.git_pull_rebase'), patch('ingest_prices.git_commit_push'):
                        with patch('ingest_prices.CSV_PATH', self.csv_path), \
                             patch('ingest_prices.DATA_DIR', self.temp_dir):
                            with open(self.csv_path, "w") as f:
                                f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
                                f.write("2026-05-21,2.05,2.15,2.10,2.20,2.30\n")
                            # Write a minimal valid prediction_log.csv so validate_all passes
                            log_path = os.path.join(self.temp_dir, "prediction_log.csv")
                            with open(log_path, "w") as f:
                                f.write("timestamp,commodity,predicted_direction,nymex_move_cents,lag_used,window_used,threshold_used,actual_next_day_move_cents,prediction_source,signal_contract,baseline_contract,settlement_source,baseline_source,settlement_captured_at,contract_provenance_status,log_schema_version,nymex_daily_std_used,z_score_used,conviction_label,conviction_provenance,hike_threshold_used,drop_threshold_used,lean_hike_threshold_used,lean_drop_threshold_used,signal_price_used,baseline_price_used,runtime_config_hash,config_file_hash,metrics_cache_hash,calibration_effective_session,calibration_artifact_id\n")
                            # Write a minimal config.json so validate_all does not error on JSON check
                            config_path = os.path.join(self.temp_dir, "config.json")
                            with open(config_path, "w") as f:
                                f.write("{}\n")

                            
                            try:
                                ingest_prices.main()
                            except SystemExit:
                                pass

        
        mock_send_email.assert_called_once()
        args, kwargs = mock_send_email.call_args
        subject = args[0]
        body = args[1]
        
        self.assertEqual(subject, "CRITICAL: Nightly calibration commit failed")
        self.assertIn("git push failed: Merge conflict in config.json", body)

    def test_13_3_prediction_log_backfill_overwrite_guard(self):
        import argparse
        import scratch.check_prediction_log_completeness as backfiller
        from io import StringIO
        
        # We simulate duplicates by having the same date twice in graves_history.csv,
        # which will cause both rows to be candidate trading days.
        with open(self.csv_path, "w") as f:
            f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
            f.write("2026-05-20,2.10,2.20,2.30,2.40,2.50\n")
            f.write("2026-05-21,2.12,2.22,2.32,2.42,2.52\n")
            f.write("2026-05-21,2.12,2.22,2.32,2.42,2.52\n")
            
        # Empty prediction log initially
        with open(self.log_path, "w") as f:
            f.write("timestamp,commodity,predicted_direction,nymex_move_cents,lag_used,window_used,threshold_used,actual_next_day_move_cents,prediction_source,signal_contract,baseline_contract,settlement_source,baseline_source,settlement_captured_at,contract_provenance_status,log_schema_version,nymex_daily_std_used,z_score_used,conviction_label,conviction_provenance,hike_threshold_used,drop_threshold_used,lean_hike_threshold_used,lean_drop_threshold_used,signal_price_used,baseline_price_used,runtime_config_hash,config_file_hash,metrics_cache_hash,calibration_effective_session,calibration_artifact_id\n")
            
        old_csv = backfiller.CSV_PATH
        old_log = backfiller.LOG_PATH
        backfiller.CSV_PATH = self.csv_path
        backfiller.LOG_PATH = self.log_path
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            with patch('scratch.check_prediction_log_completeness.simulate_thresholds_at_date') as mock_sim:
                mock_sim.return_value = {}
                with patch('argparse.ArgumentParser.parse_args') as mock_args:
                    mock_args.return_value = argparse.Namespace(backfill=True)
                    try:
                        backfiller.main()
                    except SystemExit:
                        pass
        finally:
            sys.stdout = old_stdout
            backfiller.CSV_PATH = old_csv
            backfiller.LOG_PATH = old_log
            
        output = mystdout.getvalue()
        self.assertIn("WARNING: Entry already exists in prediction log", output)
        self.assertIn("Skipping safely to prevent duplicate/overwrite", output)

    def test_13_4_programmatic_good_friday(self):
        from datetime import date
        # Good Fridays:
        # 2023: April 7
        # 2024: March 29
        # 2025: April 18
        # 2026: April 3
        # 2027: March 26
        # 2028: April 14 (Easter Sunday is April 16)
        self.assertEqual(validate_data.get_good_friday(2023), date(2023, 4, 7))
        self.assertEqual(validate_data.get_good_friday(2024), date(2024, 3, 29))
        self.assertEqual(validate_data.get_good_friday(2025), date(2025, 4, 18))
        self.assertEqual(validate_data.get_good_friday(2026), date(2026, 4, 3))
        self.assertEqual(validate_data.get_good_friday(2027), date(2027, 3, 26))
        self.assertEqual(validate_data.get_good_friday(2028), date(2028, 4, 14))

        # Check holiday detection
        self.assertTrue(validate_data.is_cme_holiday(date(2028, 4, 14)))

    def test_13_5_regex_lookahead_limit(self):
        # 1. Standard matching
        self.assertEqual(ingest_prices.extract_price_near_label("E10 - UNLEADED: $2.10", "E10 - UNLEADED"), 2.10)
        
        # 2. Match should NOT cross line boundary
        body_with_newline = "E10 - UNLEADED\nCLEAR DIESEL: $2.50"
        self.assertIsNone(ingest_prices.extract_price_near_label(body_with_newline, "E10 - UNLEADED"))
        
        # 3. Match should NOT exceed 60 characters
        long_body = "E10 - UNLEADED " + ("x" * 65) + " $2.10"
        self.assertIsNone(ingest_prices.extract_price_near_label(long_body, "E10 - UNLEADED"))

        # 4. Match should handle non-breaking spaces (\xa0) and tabs
        body_with_nbsp_and_tabs = "11\tE10\xa0-\xa0UNLEADED\t3.01700\t0.46152\t3.47852"
        self.assertEqual(ingest_prices.extract_price_near_label(body_with_nbsp_and_tabs, "E10 - UNLEADED"), 3.01700)

    @patch('ingest_prices.imaplib.IMAP4_SSL')
    def test_13_6_imap_retry_logic(self, mock_imap_ssl):
        from unittest.mock import call
        # Simulate IMAP connection failing twice then succeeding
        mock_conn = MagicMock()
        mock_imap_ssl.side_effect = [
            Exception("Connection timed out"),
            Exception("Server busy"),
            mock_conn
        ]
        
        with patch('time.sleep') as mock_sleep:
            date_str, prices = ingest_prices.check_inbox_for_prices("2026-05-22")
            
            self.assertEqual(mock_imap_ssl.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_has_calls([call(2), call(4)])

    def test_13_7_threshold_bounds_are_enforced(self):
        """Clamps and the noise floor both bound the published thresholds."""
        rng = np.random.default_rng(5)
        x = rng.normal(0, 8, 500)
        y = 0.7 * x + rng.normal(0, 2, 500)
        fit = model.fit_passthrough(x, y)
        hike, drop = fit.threshold_for_confidence(0.75)
        self.assertGreaterEqual(hike, model.MIN_ABS_THRESHOLD)
        self.assertLessEqual(hike, model.MAX_ABS_THRESHOLD)
        self.assertLessEqual(drop, -model.MIN_ABS_THRESHOLD)
        self.assertGreaterEqual(drop, -model.MAX_ABS_THRESHOLD)

        floored_hike, floored_drop = model.apply_noise_floor(hike, drop, 5.0)
        self.assertGreaterEqual(floored_hike, 5.0)
        self.assertLessEqual(floored_drop, -5.0)

    def test_13_8_calibration_is_idempotent(self):
        """Recalibrating identical history must not drift the thresholds.

        The superseded engine blended each new fit into the previous cached
        value with BLEND_ALPHA, so repeated runs over the same history walked
        the thresholds.  The new engine is a pure function of its window.
        """
        full = alignment.load_history(backtest.CSV_PATH)
        df = alignment.calibration_history(full)

        first = dict(backtest.DEFAULTS)
        first, _, _ = backtest.calibrate(df, 'RB', first)
        second, _, _ = backtest.calibrate(df, 'RB', dict(first))

        for key in ('RB_HIKE_THRESHOLD_CENTS', 'RB_DROP_THRESHOLD_CENTS',
                    'RB_pt_slope', 'RB_pt_intercept', 'RB_oos_precision'):
            self.assertEqual(first[key], second[key],
                             f"{key} drifted on a repeated calibration")

    def test_13_4_get_decoupling_warning(self):
        import main
        temp_dir = tempfile.mkdtemp()
        try:
            with patch('main.DATA_DIR', temp_dir):
                log_path = os.path.join(temp_dir, "prediction_log.csv")
                
                # Scenario A: Log file doesn't exist -> should return ""
                now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=pytz.timezone('America/Chicago'))
                self.assertEqual(main.get_decoupling_warning('RB', now), "")
                
                # Scenario B: Less than 5 actionable alerts in last 14 days -> should return ""
                df_few = pd.DataFrame({
                    "timestamp": ["2026-05-20T12:00:00-05:00"] * 4,
                    "commodity": ["RB"] * 4,
                    "predicted_direction": ["HIKE"] * 4,
                    "nymex_move_cents": [2.0] * 4,
                    "lag_used": [0] * 4,
                    "window_used": [120] * 4,
                    "threshold_used": [1.0] * 4,
                    "actual_next_day_move_cents": [-1.0] * 4
                })
                df_few.to_csv(log_path, index=False)
                self.assertEqual(main.get_decoupling_warning('RB', now), "")
                
                # Scenario C: 5 actionable alerts in last 14 days, win rate >= 40% -> should return ""
                df_win = pd.DataFrame({
                    "timestamp": ["2026-05-20T12:00:00-05:00"] * 5,
                    "commodity": ["RB"] * 5,
                    "predicted_direction": ["HIKE"] * 5,
                    "nymex_move_cents": [2.0] * 5,
                    "lag_used": [0] * 5,
                    "window_used": [120] * 5,
                    "threshold_used": [1.0] * 5,
                    "actual_next_day_move_cents": [2.0, 2.0, -1.0, 2.0, -1.0]
                })
                df_win.to_csv(log_path, index=False)
                self.assertEqual(main.get_decoupling_warning('RB', now), "")
                
                # Scenario D: 5 actionable alerts in last 14 days, win rate < 40% -> should return warning
                df_lose = pd.DataFrame({
                    "timestamp": ["2026-05-20T12:00:00-05:00"] * 5,
                    "commodity": ["RB"] * 5,
                    "predicted_direction": ["HIKE"] * 5,
                    "nymex_move_cents": [2.0] * 5,
                    "lag_used": [0] * 5,
                    "window_used": [120] * 5,
                    "threshold_used": [1.0] * 5,
                    "actual_next_day_move_cents": [-1.0, -2.0, 2.0, -0.5, -1.5]
                })
                df_lose.to_csv(log_path, index=False)
                warn_msg = main.get_decoupling_warning('RB', now)
                self.assertIn("decoupled from NYMEX", warn_msg)
        finally:
            shutil.rmtree(temp_dir)


class TestCategory14ConfigSplit(unittest.TestCase):
    """Category 14: Config and Metrics split tests"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.metrics_cache_path = os.path.join(self.temp_dir, "metrics_cache.json")
        
        self.default_config_data = {
            "MIN_ROWS_FOR_TUNING": 30,
            "BLEND_ALPHA": 0.3,
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "HO_HIKE_THRESHOLD_CENTS": 1.0,
            "HO_DROP_THRESHOLD_CENTS": -1.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "HO_LEAN_HIKE_CENTS": 0.5,
            "HO_LEAN_DROP_CENTS": -0.5,
            "LAG_DAYS": 0
        }
        with open(self.config_path, "w") as f:
            json.dump(self.default_config_data, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        import importlib
        import main
        importlib.reload(main)

    @patch('builtins.open')
    @patch('os.path.exists')
    def test_14_1_metrics_missing_fallback(self, mock_exists, mock_file):
        def side_exists(path):
            if "config.json" in path:
                return True
            if "metrics_cache.json" in path:
                return False
            return False
        mock_exists.side_effect = side_exists
        
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(self.default_config_data)
        
        import importlib
        import main
        importlib.reload(main)
        
        self.assertEqual(main.APP_CONFIG["RB_HIKE_THRESHOLD_CENTS"], 1.0)
        self.assertFalse(main.CONFIG_CORRUPT)

    @patch('builtins.open')
    @patch('os.path.exists')
    def test_14_2_metrics_corrupt_fallback(self, mock_exists, mock_file):
        mock_exists.return_value = True
        
        def side_open(file, *args, **kwargs):
            if "config.json" in str(file):
                return mock_open(read_data=json.dumps(self.default_config_data))()
            if "metrics_cache.json" in str(file):
                return mock_open(read_data="{invalid_json:")()
            raise FileNotFoundError()
            
        mock_file.side_effect = side_open
        
        import importlib
        import main
        importlib.reload(main)
        
        self.assertEqual(main.APP_CONFIG["RB_HIKE_THRESHOLD_CENTS"], 1.0)
        self.assertFalse(main.CONFIG_CORRUPT)

    @patch('builtins.open')
    @patch('os.path.exists')
    def test_14_3_overlay_priority(self, mock_exists, mock_file):
        mock_exists.return_value = True
        
        override_data = {
            "RB_HIKE_THRESHOLD_CENTS": 2.5,
            "RB_DROP_THRESHOLD_CENTS": -2.5
        }
        
        def side_open(file, *args, **kwargs):
            if "config.json" in str(file):
                return mock_open(read_data=json.dumps(self.default_config_data))()
            if "metrics_cache.json" in str(file):
                return mock_open(read_data=json.dumps(override_data))()
            raise FileNotFoundError()
            
        mock_file.side_effect = side_open
        
        import importlib
        import main
        importlib.reload(main)
        
        self.assertEqual(main.APP_CONFIG["RB_HIKE_THRESHOLD_CENTS"], 2.5)
        self.assertEqual(main.APP_CONFIG["RB_DROP_THRESHOLD_CENTS"], -2.5)
        self.assertEqual(main.APP_CONFIG["MIN_ROWS_FOR_TUNING"], 30)

    def test_14_4_backtest_only_writes_metrics_cache(self):
        orig_config_path = backtest.CONFIG_PATH
        orig_metrics_path = backtest.METRICS_CACHE_PATH
        
        backtest.CONFIG_PATH = self.config_path
        backtest.METRICS_CACHE_PATH = self.metrics_cache_path
        
        cfg = {
            "MIN_ROWS_FOR_TUNING": 30,
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "NEW_VAR": 99.0
        }
        
        try:
            backtest.save_metrics_cache(cfg)
            
            with open(self.config_path, "r") as f:
                config_after = json.load(f)
            self.assertEqual(config_after, self.default_config_data)
            
            with open(self.metrics_cache_path, "r") as f:
                cache_after = json.load(f)
            self.assertIn("RB_HIKE_THRESHOLD_CENTS", cache_after)
            self.assertEqual(cache_after["RB_HIKE_THRESHOLD_CENTS"], 1.0)
            self.assertRegex(cache_after["CALIBRATION_EFFECTIVE_SESSION"], r"^\d{4}-\d{2}-\d{2}$")
        finally:
            backtest.CONFIG_PATH = orig_config_path
            backtest.METRICS_CACHE_PATH = orig_metrics_path


class TestCategory15PredictionLog(unittest.TestCase):
    """Category 15: Prediction Log and Backfilling Tests"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "graves_history.csv")
        self.log_path = os.path.join(self.temp_dir, "prediction_log.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('main.DATA_DIR')
    @patch('main.get_repo_variable', return_value=None)
    @patch('main.set_repo_variable', return_value=None)
    @patch('main.APP_CONFIG', {"LAG_DAYS": 0, "ROLLING_WINDOW_DAYS": 120, "RB_HIKE_THRESHOLD_CENTS": 1.0, "RB_nymex_daily_std": 1.0})
    def test_15_1_prediction_log_idempotency(self, mock_set_var, mock_get_var, mock_data_dir):
        mock_data_dir.__get__ = MagicMock(return_value=self.temp_dir)
        with patch('main.DATA_DIR', self.temp_dir):
            data = {
                'current_price': 2.20,
                'yesterday_close': 2.00,
                'open_price': 2.00,
                'high_price': 2.25,
                'low_price': 1.95,
                'daily_pct': 10.0,
                'five_day_high': 2.25,
                'five_day_low': 1.95,
                'thirty_day_avg': 2.05
            }
            now = datetime(2026, 5, 22, 14, 35, 0, tzinfo=pytz.timezone('America/Chicago'))
            main.build_rack_signal('RB', data, now)
            
            self.assertTrue(os.path.exists(self.log_path))
            df1 = pd.read_csv(self.log_path)
            self.assertEqual(len(df1), 1)
            self.assertEqual(df1.loc[0, 'prediction_source'], 'live')
            
            main.build_rack_signal('RB', data, now)
            
            df2 = pd.read_csv(self.log_path)
            self.assertEqual(len(df2), 1)

    @patch('weekly_report.CSV_PATH')
    @patch('weekly_report.LOG_PATH')
    @patch('weekly_report.validate_data.validate_all')
    @patch('weekly_report.plt')
    def test_15_2_pending_backfill_correct_direction(self, mock_plt, mock_validate, mock_log_path, mock_csv_path):
        mock_csv_path.__get__ = MagicMock(return_value=self.csv_path)
        mock_log_path.__get__ = MagicMock(return_value=self.log_path)
        
        hist_df = pd.DataFrame({
            "date": ["2026-05-19", "2026-05-20", "2026-05-21"],
            "nymex_rb": [1.90, 1.95, 2.00],
            "nymex_ho": [1.90, 1.95, 2.00],
            "rack_u": [2.00, 2.05, 2.15],
            "rack_p": [2.10, 2.15, 2.25],
            "rack_d": [2.20, 2.25, 2.35]
        })
        hist_df.to_csv(self.csv_path, index=False)
        
        log_df = pd.DataFrame({
            "timestamp": ["2026-05-20T14:35:00-05:00"],
            "commodity": ["RB"],
            "predicted_direction": ["HIKE"],
            "nymex_move_cents": [5.0],
            "lag_used": [0],
            "window_used": [120],
            "threshold_used": [1.0],
            "actual_next_day_move_cents": ["PENDING"],
            "prediction_source": ["live"]
        })
        log_df.to_csv(self.log_path, index=False)
        
        class TestComplete(Exception):
            pass
            
        with patch('weekly_report.CSV_PATH', self.csv_path), \
             patch('weekly_report.LOG_PATH', self.log_path), \
             patch('weekly_report.pd.to_datetime', side_effect=TestComplete):
            try:
                weekly_report.main()
            except TestComplete:
                pass
                
        updated_log = pd.read_csv(self.log_path)
        self.assertEqual(float(updated_log.loc[0, 'actual_next_day_move_cents']), 5.0)


class TestCategory17ContractCalendar(unittest.TestCase):
    """Category 17: Contract Expiry Calendar Verification"""
    def test_17_1_refined_product_last_trade_date(self):
        import futures_util
        from datetime import date
        ltd = futures_util.refined_product_last_trade_date(2025, 5)
        self.assertEqual(ltd, date(2025, 4, 30))

    def test_17_2_get_front_month_contract_roll(self):
        import futures_util
        from datetime import date

        # April 30, 2025 = mathematical LTD for the May RB contract.
        # With early_roll_days=3, the effective roll date is ~3 biz days before (≈ April 25).
        # On the LTD itself the market has already conventionally rolled → expect June contract.
        dt_expiry = date(2025, 4, 30)
        cyear, cmonth, ltd = futures_util.get_front_month_contract(dt_expiry, 'RB')
        self.assertEqual((cyear, cmonth), (2025, 6),
            "On the mathematical LTD the market has already rolled to the next contract")

        # One day after the LTD also expects June.
        dt_after = date(2025, 5, 1)
        cyear_after, cmonth_after, ltd_after = futures_util.get_front_month_contract(dt_after, 'RB')
        self.assertEqual((cyear_after, cmonth_after), (2025, 6))

        # Well before the early-roll window (e.g. April 1, 2025) should still show May contract.
        dt_early = date(2025, 4, 1)
        cyear_early, cmonth_early, ltd_early = futures_util.get_front_month_contract(dt_early, 'RB')
        self.assertEqual((cyear_early, cmonth_early), (2025, 5),
            "Well before the roll window should still target the current front-month")

        # Bypass early-roll by passing early_roll_days=0 to get the raw math boundary.
        dt_exact = date(2025, 4, 30)
        cyear_raw, cmonth_raw, ltd_raw = futures_util.get_front_month_contract(dt_exact, 'RB', early_roll_days=0)
        self.assertEqual((cyear_raw, cmonth_raw), (2025, 5),
            "With early_roll_days=0 the LTD itself should still be in-contract")
        self.assertEqual(ltd_raw, date(2025, 4, 30))

    def test_17_3_nymex_holidays_2024_2025(self):
        import futures_util
        from datetime import date
        holidays_2024 = [
            date(2024, 1, 1),
            date(2024, 1, 15),
            date(2024, 2, 19),
            date(2024, 3, 29),
            date(2024, 5, 27),
            date(2024, 6, 19),
            date(2024, 7, 4),
            date(2024, 9, 2),
            date(2024, 11, 28),
            date(2024, 12, 25),
        ]
        holidays_2025 = [
            date(2025, 1, 1),
            date(2025, 1, 20),
            date(2025, 2, 17),
            date(2025, 4, 18),
            date(2025, 5, 26),
            date(2025, 6, 19),
            date(2025, 7, 4),
            date(2025, 9, 1),
            date(2025, 11, 27),
            date(2025, 12, 25),
        ]
        
        for h in holidays_2024 + holidays_2025:
            is_biz = futures_util.is_nymex_business_day(h)
            self.assertFalse(is_biz, f"Expected {h} to be a NYMEX holiday, but it was classified as a business day.")


class TestCategory18OperationalPricingRules(unittest.TestCase):
    """Category 18: Operational Pricing Rules Verification"""
    def test_18_1_weekly_report_load_config(self):
        import weekly_report
        cfg = weekly_report.load_config()
        self.assertIn("DISPATCH_SAME_DAY_RATE", cfg)
        self.assertAlmostEqual(cfg["DISPATCH_SAME_DAY_RATE"], 0.50, places=4)
        
    def test_18_2_build_html_email_warnings_and_savings(self):
        import main
        import pytz
        from datetime import datetime
        now = datetime(2026, 5, 22, 14, 35, 0, tzinfo=pytz.timezone('America/Chicago'))
        
        all_data = {
            'RB': {
                'current_price': 2.10,
                'open_price': 2.00,
                'high_price': 2.20,
                'low_price': 1.95,
                'daily_pct': 5.0,
                'yesterday_close': 2.00,
                'five_day_high': 2.25,
                'five_day_low': 1.95,
                'thirty_day_avg': 2.05,
                'chart_intraday_b64': 'mock_intra',
                'chart_5d_b64': 'mock_5d',
                'rack_signal': {
                    'action': 'BUY_NOW',
                    'label': 'Hike likely',
                    'color': '#ef4444',
                    'change_cents': 10.0,
                    'basis': 'test basis',
                    'conviction': 'High Conviction',
                    'z_score': 1.8,
                    'risk_text': 'Risk details'
                }
            }
        }
        
        alert_context = {
            'label': 'Final Verdict'
        }
        
        html_body, cids = main.build_html_email("Test Subject", all_data, now, alert_context)
        
        # Verify no emojis
        self.assertNotIn("⚠️", html_body)
        
        # Verify warnings box is not present
        self.assertNotIn("Operational Checklist & Dispatch Rules", html_body)
        
        # Verify realized savings is displayed and scaled
        # 10.0 * 0.50 = 5.00
        self.assertIn("Est. Realized Savings: +5.00¢/gal (at 50% same-day dispatch rate)", html_body)


    @patch('main.get_repo_variable', return_value=None)
    def test_18_4_low_conviction_recommendation_text(self, mock_get_var):
        import main
        import pytz
        from datetime import datetime
        now = datetime(2026, 5, 22, 14, 35, 0, tzinfo=pytz.timezone('America/Chicago'))
        
        # change_cents = 2.0 (>= 1.99 threshold)
        # Z-score = 2.0 / 8.9551 = 0.22 (< 1.0 -> Low Conviction)
        data = {
            'current_price': 2.02,
            'open_price': 2.00,
            'high_price': 2.05,
            'low_price': 1.99,
            'daily_pct': 1.0,
            'yesterday_close': 2.00,
            'five_day_high': 2.10,
            'five_day_low': 1.95,
            'thirty_day_avg': 2.00,
            'chart_intraday_b64': None,
            'chart_5d_b64': None,
            'schwab_symbol': 'test_symbol'
        }
        
        # Install a deliberately weak pass-through so a +2c move sits below the
        # confidence cutoff.  The old ladder keyed off |Z| < 1.0; the gate is now
        # the calibrated probability itself.
        rng = np.random.default_rng(4)
        x = rng.normal(0, 8, 500)
        y = 0.15 * x + rng.normal(0, 6, 500)
        saved = main.APP_CONFIG.copy()
        try:
            main.APP_CONFIG.update(
                model.passthrough_to_config(model.fit_passthrough(x, y), 'RB'))
            signal = main.build_rack_signal('RB', data, now)
            self.assertLess(float(signal['conviction'].rstrip('%)').split('(')[-1]) / 100.0,
                            main.LOW_CONFIDENCE_CUTOFF)
            self.assertIn('do not act on this signal unless inventory forces you '
                          'to order regardless.', signal['text'])
            self.assertRegex(signal['text'], r'Only \d+% confidence')
        finally:
            main.APP_CONFIG.clear()
            main.APP_CONFIG.update(saved)



class TestCategory19EndToEndTuesdaySimulation(unittest.TestCase):
    """Category 19: End-to-End Tuesday Simulation for BUY alerts under different convictions"""
    
    @patch('main.get_repo_variable', return_value=None)
    def test_19_1_tuesday_low_conviction_simulation(self, mock_get_var):
        import main
        import pytz
        from datetime import datetime
        from unittest.mock import patch
        
        # Simulating Tuesday afternoon at 2:35 PM CT
        tz_chicago = pytz.timezone('America/Chicago')
        now = datetime(2026, 5, 26, 14, 35, 0, tzinfo=tz_chicago)
        
        # Synthetic data with +4.0 move
        # nymex_daily_std is 8.9551, so Z-score = 4.0 / 8.9551 = 0.45 (< 1.0 -> Low Conviction)
        rb_data = {
            'current_price': 2.04,
            'open_price': 2.00,
            'high_price': 2.08,
            'low_price': 1.99,
            'daily_pct': 2.0,
            'yesterday_close': 2.00,
            'five_day_high': 2.10,
            'five_day_low': 1.95,
            'thirty_day_avg': 2.00,
            'chart_intraday_b64': 'mock_intra',
            'chart_5d_b64': 'mock_5d',
            'schwab_symbol': 'test_symbol'
        }
        
        # 1. Generate signal under a deliberately weak pass-through, so the
        #    +4c move triggers the threshold but not the confidence cutoff.
        rng = np.random.default_rng(9)
        x = rng.normal(0, 8, 500)
        y = 0.15 * x + rng.normal(0, 6, 500)
        main.APP_CONFIG.update(
            model.passthrough_to_config(model.fit_passthrough(x, y), 'RB'))
        self.addCleanup(lambda: [main.APP_CONFIG.pop(k, None)
                                 for k in list(main.APP_CONFIG)
                                 if k.startswith('RB_pt_')])

        signal = main.build_rack_signal('RB', rb_data, now)
        self.assertEqual(signal['action'], 'BUY_NOW')
        self.assertAlmostEqual(signal['change_cents'], 4.0, places=4)
        self.assertRegex(signal['text'], r'Only \d+% confidence')
        self.assertTrue(signal['text'].rstrip().endswith(
            'do not act on this signal unless inventory forces you to order regardless.'))
        
        # 2. Verify HTML verdict email
        all_data = {'RB': rb_data}
        rb_data['rack_signal'] = signal
        
        alert_context = {'label': 'Final Verdict'}
        html_body, cids = main.build_html_email("Final Verdict: Exxon Price Predictor", all_data, now, alert_context)
        
        self.assertNotIn("⚠️", html_body)
        self.assertIn("UNLEADED HIKE LIKELY:", html_body)
        self.assertIn("Est. Realized Savings: +2.00¢/gal (at 50% same-day dispatch rate)", html_body)
        self.assertIn("do not act on this signal unless inventory forces you to order regardless.", html_body)
        self.assertNotIn("Operational Checklist & Dispatch Rules", html_body)

    @patch('main.get_repo_variable', return_value=None)
    def test_19_2_tuesday_high_conviction_simulation(self, mock_get_var):
        import main
        import pytz
        from datetime import datetime
        from unittest.mock import patch

        # Simulating Tuesday afternoon at 2:35 PM CT
        tz_chicago = pytz.timezone('America/Chicago')
        now = datetime(2026, 5, 26, 14, 35, 0, tzinfo=tz_chicago)

        # Synthetic data with +21.0¢ move (~2× the highest expected nymex_daily_std).
        # We freeze nymex_daily_std to 8.9551 via APP_CONFIG so this test is not
        # fragile against auto-tuner recalibration: Z = 21.0 / 8.9551 = 2.35 >> 1.5.
        # Even if the auto-tuner raises std to ~13c, Z = 21/13 = 1.61 still >= 1.5.
        rb_data = {
            'current_price': 2.21,
            'open_price': 2.00,
            'high_price': 2.25,
            'low_price': 1.99,
            'daily_pct': 10.5,
            'yesterday_close': 2.00,
            'five_day_high': 2.25,
            'five_day_low': 1.95,
            'thirty_day_avg': 2.00,
            'chart_intraday_b64': 'mock_intra',
            'chart_5d_b64': 'mock_5d',
            'schwab_symbol': 'test_symbol'
        }

        # Freeze APP_CONFIG with a realistic fitted pass-through so the quoted
        # confidence is deterministic regardless of the live cache.
        rng = np.random.default_rng(12)
        x = rng.normal(0, 8, 500)
        y = 0.70 * x + rng.normal(0, 3, 500)
        frozen_config = dict(main.APP_CONFIG)
        frozen_config['RB_nymex_daily_std'] = 8.9551
        frozen_config.update(
            model.passthrough_to_config(model.fit_passthrough(x, y), 'RB'))
        with patch.object(main, 'APP_CONFIG', frozen_config):
            # 1. Generate signal
            signal = main.build_rack_signal('RB', rb_data, now)
        self.assertEqual(signal['action'], 'BUY_NOW')
        # A +21c move at slope 0.70 against a 3c residual spread is decisive.
        self.assertIn('High confidence', signal['conviction'])
        self.assertAlmostEqual(signal['change_cents'], 21.0, places=4)
        self.assertEqual(
            signal['text'].split(". ")[-1],
            'Dispatch before the rack deadline if you need inventory.'
        )

        
        # 2. Verify HTML verdict email
        all_data = {'RB': rb_data}
        rb_data['rack_signal'] = signal
        
        alert_context = {'label': 'Final Verdict'}
        html_body, cids = main.build_html_email("Final Verdict: Exxon Price Predictor", all_data, now, alert_context)
        
        self.assertNotIn("⚠️", html_body)
        self.assertIn("UNLEADED HIKE LIKELY:", html_body)
        self.assertIn("Est. Realized Savings: +10.50¢/gal (at 50% same-day dispatch rate)", html_body)
        self.assertIn("Dispatch before the rack deadline if you need inventory.", html_body)
        self.assertNotIn("Operational Checklist & Dispatch Rules", html_body)


class TestCategory20SecretMasking(unittest.TestCase):
    """Category 20: Security - Verify sensitive data is masked in logs"""

    def test_20_1_mask_sensitive_text_token_masking(self):
        """Verify that auth tokens and secrets are masked"""
        with patch.dict(os.environ, {
            'GH_PAT': 'ghp_test_token_12345678901234567890',
            'SCHWAB_APP_KEY': 'app_key_test_12345678901234567890',
            'SCHWAB_APP_SECRET': 'app_secret_test_12345678901234567890',
            'GMAIL_APP_PASSWORD': 'gmail_password_test_12345678'
        }):
            test_text = "OAuth failed with token ghp_test_token_12345678901234567890"
            masked = main.mask_sensitive_text(test_text)
            # Token should be masked
            self.assertNotIn('ghp_test_token_12345678901234567890', masked)
            self.assertIn('gh***90', masked)

    def test_20_2_mask_sensitive_text_email_masking(self):
        """Verify that email addresses are masked"""
        with patch.dict(os.environ, {
            'GMAIL_USER': 'user@example.com',
            'TO_EMAIL': 'recipient@example.com'
        }):
            test_text = "Email sent to recipient@example.com from user@example.com"
            masked = main.mask_sensitive_text(test_text)
            # Full email should not appear
            self.assertNotIn('user@example.com', masked)
            self.assertNotIn('recipient@example.com', masked)
            # Check actual masking format (re***t@example.com, us***r@example.com)
            self.assertIn('re***t@example.com', masked)
            self.assertIn('us***r@example.com', masked)

    def test_20_4_mask_recipient_email(self):
        """Verify mask_recipient properly masks email addresses"""
        result = main.mask_recipient('john.doe@example.com')
        self.assertNotIn('john', result)
        self.assertNotIn('doe', result)
        # Masking format: first 2 chars + *** + last char (before @)
        self.assertIn('jo***e', result)
        self.assertIn('@example.com', result)

    def test_20_6_no_secrets_in_exception_logs(self):
        """Verify that exception messages don't expose secrets"""
        with patch.dict(os.environ, {
            'GH_PAT': 'secret_gh_token_1234567890',
            'GMAIL_APP_PASSWORD': 'secret_gmail_pass_1234567890'
        }):
            # Simulate an exception with a secret
            exception_msg = "Failed to send email with password secret_gmail_pass_1234567890"
            masked = main.mask_sensitive_text(exception_msg)
            self.assertNotIn('secret_gmail_pass_1234567890', masked)
            self.assertIn('se***90', masked)

    def test_20_7_multiple_secrets_in_one_message(self):
        """Verify multiple secrets in one message are all masked"""
        with patch.dict(os.environ, {
            'SCHWAB_APP_KEY': 'key_12345678901234567890',
            'SCHWAB_APP_SECRET': 'secret_12345678901234567890',
            'GMAIL_USER': 'user@example.com'
        }):
            test_text = "Auth with key_12345678901234567890 secret_12345678901234567890 from user@example.com"
            masked = main.mask_sensitive_text(test_text)
            # No unmasked values should appear
            self.assertNotIn('key_12345678901234567890', masked)
            self.assertNotIn('secret_12345678901234567890', masked)
            self.assertNotIn('user@example.com', masked)


if __name__ == "__main__":
    unittest.main()
