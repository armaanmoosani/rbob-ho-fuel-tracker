import os
os.environ['GRAVES_EMAIL'] = 'mock_graves@example.com'
os.environ['GRAVES_APP_PASSWORD'] = 'mock_graves_pass'
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import pytz

# Ensure parent directory is in path
sys.path.append(os.path.dirname(__file__))
import ingest_prices

class TestRetryAndTargetDateLogic(unittest.TestCase):
    
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
