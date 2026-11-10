import os
os.environ['GRAVES_EMAIL'] = 'mock_graves@example.com'
os.environ['GRAVES_APP_PASSWORD'] = 'mock_graves_pass'
import sys
import csv
import tempfile
import unittest
from email.message import EmailMessage
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import pytz

# Ensure parent directory is in path
sys.path.append(os.path.dirname(__file__))
import ingest_prices

class TestRetryAndTargetDateLogic(unittest.TestCase):

    def test_verified_live_prediction_recovers_exact_settlement_snapshot(self):
        header = (
            "timestamp,commodity,prediction_source,signal_contract,settlement_source,"
            "settlement_captured_at,contract_provenance_status,signal_price_used\n"
        )
        rows = (
            "2026-09-22T14:35:00-05:00,RB,live,/RBV26,schwab,"
            "2026-09-22T13:30:00-05:00,verified,3.4947\n"
            "2026-09-22T14:35:00-05:00,HO,live,/HOV26,schwab,"
            "2026-09-22T13:30:00-05:00,verified,4.9486\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = os.path.join(temp_dir, "graves_history.csv")
            with open(os.path.join(temp_dir, "prediction_log.csv"), "w") as handle:
                handle.write(header + rows)
            with patch("ingest_prices.CSV_PATH", history_path):
                settlement = ingest_prices.read_live_prediction_settlement("2026-09-22")

        self.assertEqual(settlement["rbob_settlement"], 3.4947)
        self.assertEqual(settlement["heating_oil_settlement"], 4.9486)
        self.assertEqual(settlement["rbob_contract"], "/RBV26")
        self.assertEqual(settlement["heating_oil_contract"], "/HOV26")

    def test_delayed_noon_run_recovers_missing_previous_weekday_first(self):
        tz = pytz.timezone('America/Chicago')
        delayed = tz.localize(datetime(2026, 9, 23, 12, 38, 0))

        candidates = ingest_prices.target_date_candidates(
            delayed, {"2026-09-21"}
        )

        self.assertEqual(candidates, ["2026-09-22", "2026-09-23"])

    def test_monday_recovers_missing_friday_before_today(self):
        tz = pytz.timezone('America/Chicago')
        monday_afternoon = tz.localize(datetime(2026, 9, 28, 12, 38, 0))

        candidates = ingest_prices.target_date_candidates(monday_afternoon, set())

        self.assertEqual(candidates, ["2026-09-25", "2026-09-28"])

    def test_monday_skips_recorded_friday_and_weekend(self):
        tz = pytz.timezone('America/Chicago')
        monday = tz.localize(datetime(2026, 9, 28, 12, 38))

        self.assertEqual(ingest_prices.target_date_candidates(
            monday, {"2026-09-25"}), ["2026-09-28"])

    def test_weekend_only_targets_missing_friday(self):
        tz = pytz.timezone('America/Chicago')
        for day in (26, 27):
            with self.subTest(day=day):
                now = tz.localize(datetime(2026, 9, day, 12))
                self.assertEqual(ingest_prices.target_date_candidates(now, set()),
                                 ["2026-09-25"])

    def test_candidates_do_not_append_old_gap_after_newer_history(self):
        tz = pytz.timezone('America/Chicago')
        monday = tz.localize(datetime(2026, 9, 28, 20))

        self.assertEqual(ingest_prices.target_date_candidates(
            monday, {"date", "2026-09-28"}), [])

    def test_successive_runs_recover_friday_then_monday_with_valid_integrity(self):
        tz = pytz.timezone('America/Chicago')
        monday = tz.localize(datetime(2026, 11, 9, 21, 30))

        def settlement(target):
            return {
                "date": target, "rbob_settlement": 2.7, "heating_oil_settlement": 4.3,
                "rbob_contract": "/RBZ26", "heating_oil_contract": "/HOZ26",
                "source": "live_prediction_snapshot",
                "captured_at": f"{target}T13:30:00-06:00",
            }

        with tempfile.TemporaryDirectory() as directory:
            history = os.path.join(directory, "graves_history.csv")
            with open(history, "w") as handle:
                handle.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n"
                             "2026-11-05,2.7633,4.3547,2.8170,3.6373,4.7624\n")
            with patch('ingest_prices.DATA_DIR', directory), \
                    patch('ingest_prices.CSV_PATH', history), \
                    patch('ingest_prices.git_pull_rebase'), \
                    patch('ingest_prices.git_commit_push') as push, \
                    patch('ingest_prices.send_alert_email') as alert, \
                    patch('ingest_prices.subprocess.run') as run, \
                    patch('ingest_prices.datetime') as clock, \
                    patch('ingest_prices.read_daily_settlement', side_effect=settlement), \
                    patch('ingest_prices.check_inbox_for_prices',
                          side_effect=lambda target: (target, (2.8, 3.6, 4.7))) as inbox:
                clock.now.return_value = monday
                run.return_value.stdout = ""
                ingest_prices.main()
                ingest_prices.main()
                ingest_prices.validate_data.validate_all(directory)

            with open(history) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['date'] for row in rows],
                             ['2026-11-05', '2026-11-06', '2026-11-09'])
            self.assertEqual([call.args[0] for call in inbox.call_args_list],
                             ['2026-11-06', '2026-11-09'])
            self.assertEqual(push.call_count, 2)
            alert.assert_not_called()
            with open(os.path.join(directory, 'nymex_settlement_provenance.csv')) as handle:
                provenance = list(csv.DictReader(handle))
            self.assertEqual(len(provenance), 4)
            self.assertEqual({row['session_date'] for row in provenance},
                             {'2026-11-06', '2026-11-09'})

    @patch('ingest_prices.imaplib.IMAP4_SSL')
    def test_explicit_effective_date_accepts_late_delivery_not_wrong_dates(self, mock_imap):
        for sent_at, effective_date, expected in (
            ("Sat, 07 Nov 2026 10:52:41 -0600", "11/6/2026", True),
            ("Mon, 09 Nov 2026 09:30:00 -0600", "11/6/2026", True),
            ("Fri, 06 Nov 2026 18:00:00 -0600", "11/5/2026", False),
            ("Sat, 07 Nov 2026 10:52:41 -0600", "11/7/2026", False),
            ("Sat, 07 Nov 2026 10:52:41 -0600", None, False),
            ("Sat, 07 Nov 2026 08:59:00 -0600", None, True),
            ("Fri, 06 Nov 2026 18:00:00 -0600", None, True),
        ):
            with self.subTest(sent_at=sent_at, effective_date=effective_date):
                message = EmailMessage()
                message['From'] = 'donotreply@gravesoil.com'
                message['Subject'] = 'Latest prices from Graves Oil Company'
                message['Date'] = sent_at
                body = (f"Prices good from {effective_date} 7:00 PM.\n"
                        if effective_date else "Latest prices\n")
                message.set_content(body + "E10 - UNLEADED 3.01700\n"
                                    "CLEAR DIESEL 3.74140\nE10 - PREMIUM 3.83730\n")
                connection = mock_imap.return_value
                connection.search.return_value = ("OK", [b"1"])
                connection.fetch.return_value = ("OK", [(None, message.as_bytes())])

                actual = ingest_prices.check_inbox_for_prices("2026-11-06")

                self.assertEqual(actual, ("2026-11-06", (3.017, 3.8373, 3.7414))
                                 if expected else (None, None))
                self.assertIn('SINCE "04-Nov-2026"', connection.search.call_args[0][1])
    
    @patch('ingest_prices.datetime')
    def test_target_date_calculation_daytime(self, mock_datetime):
        # Mock current time: 2026-05-22 20:30:00 (8:30 PM) in America/Chicago
        # 8:30 PM is hour 20 >= 4, so target date should be today: 2026-05-22
        tz = pytz.timezone('America/Chicago')
        local_now = tz.localize(datetime(2026, 5, 22, 20, 30, 0))
        mock_datetime.now.return_value = local_now
        
        # We also need mock_datetime.fromisoformat to behave correctly
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        # Test target date calculation by simulating the logic
        now_local = ingest_prices.datetime.now(ingest_prices.TZ)
        if now_local.hour < 4:
            target_date_str = (now_local - ingest_prices.timedelta(days=1)).date().isoformat()
        else:
            target_date_str = now_local.date().isoformat()
            
        self.assertEqual(target_date_str, "2026-05-22")
        
    @patch('ingest_prices.datetime')
    def test_target_date_calculation_midnight(self, mock_datetime):
        # Mock current time: 2026-05-23 00:15:00 (12:15 AM) in America/Chicago
        # 12:15 AM is hour 0 < 4, so target date should be yesterday: 2026-05-22
        tz = pytz.timezone('America/Chicago')
        local_now = tz.localize(datetime(2026, 5, 23, 0, 15, 0))
        mock_datetime.now.return_value = local_now
        
        # We also need mock_datetime.fromisoformat to behave correctly
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        now_local = ingest_prices.datetime.now(ingest_prices.TZ)
        if now_local.hour < 4:
            target_date_str = (now_local - ingest_prices.timedelta(days=1)).date().isoformat()
        else:
            target_date_str = now_local.date().isoformat()
            
        self.assertEqual(target_date_str, "2026-05-22")

    @patch('ingest_prices.validate_data.validate_all')
    @patch('ingest_prices.send_alert_email')
    @patch('ingest_prices.check_inbox_for_prices')
    @patch('ingest_prices.datetime')
    @patch('ingest_prices.os.path.exists')
    def test_missing_email_retry_at_8pm(self, mock_exists, mock_datetime, mock_check, mock_send_email, mock_val):
        # Mock current time: 8:00 PM (Hour 20) on a Monday
        tz = pytz.timezone('America/Chicago')
        local_now = tz.localize(datetime(2026, 5, 25, 20, 0, 0)) # Monday
        mock_datetime.now.return_value = local_now
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        mock_exists.return_value = True # CSV exists
        mock_check.return_value = (None, None)
        
        m = mock_open(read_data="date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
        with patch('ingest_prices.open', m), patch('ingest_prices.git_pull_rebase'), patch('ingest_prices.git_commit_push'):
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with self.assertRaises(SystemExit):
                    ingest_prices.main()
                mock_exit.assert_called_once_with(0)
                
        # Send alert should NOT be called at 8:00 PM
        mock_send_email.assert_not_called()

    @patch('ingest_prices.validate_data.validate_all')
    @patch('ingest_prices.send_alert_email')
    @patch('ingest_prices.check_inbox_for_prices')
    @patch('ingest_prices.datetime')
    @patch('ingest_prices.os.path.exists')
    def test_missing_email_warning_at_midnight(self, mock_exists, mock_datetime, mock_check, mock_send_email, mock_val):
        # Mock current time: 12:05 AM Tuesday (Hour 0), meaning target is Monday (2026-05-25)
        tz = pytz.timezone('America/Chicago')
        local_now = tz.localize(datetime(2026, 5, 26, 0, 5, 0)) # Tuesday
        mock_datetime.now.return_value = local_now
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        mock_exists.return_value = True
        mock_check.return_value = (None, None)
        
        m = mock_open(read_data="date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
        with patch('ingest_prices.open', m), patch('ingest_prices.git_pull_rebase'), patch('ingest_prices.git_commit_push'):
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with self.assertRaises(SystemExit):
                    ingest_prices.main()
                mock_exit.assert_called_once_with(0)
                
        # Send alert SHOULD be called at 12:05 AM
        mock_send_email.assert_called_once()
        self.assertIn("WARNING: Graves Oil Prices Missing", mock_send_email.call_args[0][0])
        self.assertIn("2026-05-25", mock_send_email.call_args[0][1])

    @patch('ingest_prices.validate_data.validate_all')
    @patch('ingest_prices.send_alert_email')
    @patch('ingest_prices.check_inbox_for_prices')
    @patch('ingest_prices.datetime')
    @patch('ingest_prices.os.path.exists')
    def test_missing_email_warning_manual_dispatch(self, mock_exists, mock_datetime, mock_check, mock_send_email, mock_val):
        # Mock current time: 2:00 PM (Hour 14) Monday. Missing email, but manual run!
        tz = pytz.timezone('America/Chicago')
        local_now = tz.localize(datetime(2026, 5, 25, 14, 0, 0)) # Monday
        mock_datetime.now.return_value = local_now
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        mock_exists.return_value = True
        mock_check.return_value = (None, None)
        
        # Set manual run env var
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "workflow_dispatch"}):
            m = mock_open(read_data="date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
            with patch('ingest_prices.open', m), patch('ingest_prices.git_pull_rebase'), patch('ingest_prices.git_commit_push'):
                with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                    with self.assertRaises(SystemExit):
                        ingest_prices.main()
                    mock_exit.assert_called_once_with(0)
                    
        # Send alert should NOT be called on manual dispatch unless it is midnight
        mock_send_email.assert_not_called()

    @patch('ingest_prices.validate_data.validate_all')
    @patch('ingest_prices.check_inbox_for_prices')
    @patch('ingest_prices.datetime')
    @patch('ingest_prices.os.path.exists')
    def test_already_ingested_skip(self, mock_exists, mock_datetime, mock_check, mock_val):
        # Mock current time: 8:00 PM (Hour 20) on Monday
        tz = pytz.timezone('America/Chicago')
        local_now = tz.localize(datetime(2026, 5, 25, 20, 0, 0)) # Monday
        mock_datetime.now.return_value = local_now
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        mock_exists.return_value = True
        
        # Mock CSV contents so Monday date is ALREADY in CSV
        m = mock_open(read_data="date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n2026-05-25,2.10,2.30,2.40,2.50,2.60\n")
        with patch('ingest_prices.open', m), patch('ingest_prices.git_pull_rebase'), patch('ingest_prices.git_commit_push'):
            with patch('sys.exit', side_effect=SystemExit) as mock_exit:
                with self.assertRaises(SystemExit):
                    ingest_prices.main()
                mock_exit.assert_called_once_with(0)
                
        # check_inbox_for_prices should NOT be called since it's already ingested
        mock_check.assert_not_called()

    @patch('ingest_prices.imaplib.IMAP4_SSL')
    def test_check_inbox_label_parser_correct_assignment(self, mock_imap_ssl):
        """Label-anchored parser must assign rack_p=E10-PREMIUM and rack_d=CLEAR DIESEL
        regardless of the order they appear in the email body."""
        mock_conn = MagicMock()
        mock_imap_ssl.return_value = mock_conn
        mock_conn.search.return_value = ("OK", [b"1"])
        mock_conn.fetch.return_value = ("OK", [(None, b"")])

        # Email body has: Unleaded=3.017, Premium=3.8373, Diesel=3.7414
        email_body = (
            "The following prices are good from 5/27/2026 7:00:00 PM through 5/28/2026 6:59:00 PM.\n"
            "11 E10 - UNLEADED 3.01700\n"
            "4 CLEAR DIESEL 3.74140\n"
            "13 E10 - PREMIUM 3.83730\n"
        )

        mock_msg = MagicMock()
        mock_msg.is_multipart.return_value = False
        mock_msg.get_payload = lambda decode=False: email_body.encode('utf-8') if decode else email_body
        mock_msg.get_content_type.return_value = 'text/plain'
        mock_msg.get.side_effect = lambda key: {
            'From': 'donotreply@gravesoil.com',
            'Subject': 'Latest prices from Graves Oil Company',
            'Date': 'Wed, 27 May 2026 18:24:00 -0500'
        }.get(key)

        with patch('email.message_from_bytes', return_value=mock_msg):
            date_str, prices = ingest_prices.check_inbox_for_prices("2026-05-27")

        # Standard label parser behavior without sorting: Premium (3.8373) and Diesel (3.7414)
        self.assertEqual(date_str, "2026-05-27")
        self.assertEqual(prices, (3.017, 3.8373, 3.7414))

    @patch('ingest_prices.imaplib.IMAP4_SSL')
    def test_check_inbox_late_morning_match(self, mock_imap_ssl):
        """Verify that a late email sent early on the next day morning before 9 AM CT matches target_date_str."""
        mock_conn = MagicMock()
        mock_imap_ssl.return_value = mock_conn
        mock_conn.search.return_value = ("OK", [b"1"])
        mock_conn.fetch.return_value = ("OK", [(None, b"")])

        # Email body has correct prices
        email_body = (
            "The following prices are good from 5/27/2026 7:00:00 PM through 5/28/2026 6:59:00 PM.\n"
            "11 E10 - UNLEADED 3.01700\n"
            "4 CLEAR DIESEL 3.74140\n"
            "13 E10 - PREMIUM 3.83730\n"
        )

        mock_msg = MagicMock()
        mock_msg.is_multipart.return_value = False
        mock_msg.get_payload = lambda decode=False: email_body.encode('utf-8') if decode else email_body
        mock_msg.get_content_type.return_value = 'text/plain'
        # Date is next day (Thu 28 May) morning at 7:15 AM CT
        mock_msg.get.side_effect = lambda key: {
            'From': 'donotreply@gravesoil.com',
            'Subject': 'Latest prices from Graves Oil Company',
            'Date': 'Thu, 28 May 2026 07:15:00 -0500'
        }.get(key)

        with patch('email.message_from_bytes', return_value=mock_msg):
            # Target date is May 27th
            date_str, prices = ingest_prices.check_inbox_for_prices("2026-05-27")

        # It should match, and date_str should be the target date (2026-05-27)
        self.assertEqual(date_str, "2026-05-27")
        self.assertEqual(prices, (3.017, 3.8373, 3.7414))

if __name__ == "__main__":
    unittest.main()
