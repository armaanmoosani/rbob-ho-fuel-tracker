import unittest
import threading
import json
import os
import tempfile
from unittest.mock import patch

import pandas as pd

import weekly_report
import ingest_prices


class TestPredictionSourceReporting(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-07-20T12:00:00", tz="America/Chicago")
        self.rows = pd.DataFrame(
            {
                "timestamp_dt": [
                    pd.Timestamp("2026-07-18T12:00:00", tz="America/Chicago"),
                    pd.Timestamp("2026-07-18T12:00:00", tz="America/Chicago"),
                ],
                "predicted_direction": ["HIKE", "DROP"],
                "savings_cents": [3.0, 7.0],
                "is_correct": [True, True],
            }
        )

    def test_live_and_backfill_statistics_are_separate(self):
        live = weekly_report.summarize_prediction_source(
            self.rows.iloc[[0]], self.now, 0.50
        )
        backfill = weekly_report.summarize_prediction_source(
            self.rows.iloc[[1]], self.now, 0.50
        )

        self.assertEqual(live["rows"], 1)
        self.assertEqual(live["savings_cents"], 3.0)
        self.assertEqual(backfill["rows"], 1)
        self.assertEqual(backfill["savings_cents"], 7.0)

    def test_permutation_test_requires_thirty_live_predictions(self):
        self.assertFalse(weekly_report.significance_available(29, 29))
        self.assertFalse(weekly_report.significance_available(30, 29))
        self.assertTrue(weekly_report.significance_available(30, 30))


class TestConcurrentAlertStateUpdates(unittest.TestCase):
    def test_concurrent_fetches_persist_one_merged_symbol_state(self):
        import main

        barrier = threading.Barrier(3)
        now = pd.Timestamp("2026-07-20T12:00:00", tz="America/Chicago").to_pydatetime()

        def fetch(prefix, cfg, fetch_now, access_token, alert_state):
            barrier.wait(timeout=2)
            return {
                "prefix": prefix,
                "current_price": 1.0,
                "_alert_state_updates": {
                    f"ACTIVE_SYMBOL_{prefix}_2026-07-20": f"/{prefix}N26"
                },
            }

        with patch("main.fetch_commodity", side_effect=fetch), patch("main.save_alert_state") as save_state:
            all_data, updates = main.fetch_all_commodities(now, "token", {"SENT_KEEP": "2026-07-20"})
            merged = main.merge_alert_state_updates({"SENT_KEEP": "2026-07-20"}, updates)
            main.save_alert_state(merged)

        self.assertEqual(set(all_data), {"RB", "HO", "CL"})
        self.assertEqual(merged["SENT_KEEP"], "2026-07-20")
        self.assertEqual(merged["ACTIVE_SYMBOL_RB_2026-07-20"], "/RBN26")
        self.assertEqual(merged["ACTIVE_SYMBOL_HO_2026-07-20"], "/HON26")
        self.assertEqual(merged["ACTIVE_SYMBOL_CL_2026-07-20"], "/CLN26")
        save_state.assert_called_once_with(merged)


class TestConfigurationOverlayConsistency(unittest.TestCase):
    def test_main_replay_and_statistics_prefer_metrics_cache(self):
        import main
        import replay_day
        import verify_statistics

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            metrics_path = os.path.join(temp_dir, "metrics_cache.json")
            with open(config_path, "w") as f:
                json.dump({"RB_HIKE_THRESHOLD_CENTS": 1.80}, f)
            with open(metrics_path, "w") as f:
                json.dump({"RB_HIKE_THRESHOLD_CENTS": 2.99}, f)

            main_cfg, _ = main.load_runtime_config(config_path, metrics_path)
            replay_cfg = replay_day.load_config(config_path, metrics_path)
            statistics_cfg = verify_statistics.load_config(config_path, metrics_path)

        self.assertEqual(main_cfg["RB_HIKE_THRESHOLD_CENTS"], 2.99)
        self.assertEqual(replay_cfg["RB_HIKE_THRESHOLD_CENTS"], 2.99)
        self.assertEqual(statistics_cfg["RB_HIKE_THRESHOLD_CENTS"], 2.99)


class TestContractProvenance(unittest.TestCase):
    def test_mismatched_snapshot_and_baseline_suppresses_live_signal_and_logs_identity(self):
        import main

        now = pd.Timestamp("2026-07-20T14:35:00", tz="America/Chicago").to_pydatetime()
        data = {
            "current_price": 2.20,
            "yesterday_close": 2.00,
            "schwab_symbol": "/RBN26",
            "baseline_schwab_symbol": "/RBQ26",
            "baseline_source": "schwab_close_price",
            "data_source": "schwab",
            "contract_provenance_required": True,
            "settlement_snapshot": {
                "price": 2.20,
                "schwab_symbol": "/RBN26",
                "source": "schwab",
                "captured_at": "2026-07-20T13:35:00-05:00",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch("main.DATA_DIR", temp_dir):
            signal = main.build_rack_signal("RB", data, now)
            log = pd.read_csv(os.path.join(temp_dir, "prediction_log.csv"))

        self.assertEqual(signal["action"], "NO_EDGE")
        self.assertEqual(signal["contract_provenance"]["status"], "mismatch_suppressed")
        self.assertEqual(log.loc[0, "predicted_direction"], "FLAT")
        self.assertEqual(log.loc[0, "signal_contract"], "/RBN26")
        self.assertEqual(log.loc[0, "baseline_contract"], "/RBQ26")
        self.assertEqual(log.loc[0, "contract_provenance_status"], "mismatch_suppressed")

    def test_matching_contract_allows_signal_and_marks_provenance_verified(self):
        import main

        now = pd.Timestamp("2026-07-20T14:35:00", tz="America/Chicago").to_pydatetime()
        data = {
            "current_price": 2.20,
            "yesterday_close": 2.00,
            "schwab_symbol": "/RBN26",
            "baseline_schwab_symbol": "/RBN26",
            "baseline_source": "schwab_close_price",
            "data_source": "schwab",
            "contract_provenance_required": True,
            "settlement_snapshot": {
                "price": 2.20,
                "schwab_symbol": "/RBN26",
                "source": "schwab",
                "captured_at": "2026-07-20T13:35:00-05:00",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch("main.DATA_DIR", temp_dir), patch("main.is_contract_roll_day", return_value=False):
            signal = main.build_rack_signal("RB", data, now)

        self.assertNotEqual(signal["label"], "Contract provenance unavailable")
        self.assertEqual(signal["contract_provenance"]["status"], "verified")

    def test_settlement_ledger_preserves_contract_and_is_idempotent(self):
        settlement = {
            "rbob_settlement": 2.20,
            "heating_oil_settlement": 2.30,
            "rbob_contract": "/RBN26",
            "heating_oil_contract": "/HON26",
            "rbob_yfinance_symbol": "RBN26.NYM",
            "heating_oil_yfinance_symbol": "HON26.NYM",
            "rbob_source": "schwab",
            "heating_oil_source": "schwab",
            "captured_at": "2026-07-20T13:35:00-05:00",
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ingest_prices.CSV_PATH", os.path.join(temp_dir, "graves_history.csv")
        ):
            ingest_prices.append_settlement_provenance(settlement, "2026-07-20")
            ingest_prices.append_settlement_provenance(settlement, "2026-07-20")
            ledger = pd.read_csv(os.path.join(temp_dir, "nymex_settlement_provenance.csv"))

        self.assertEqual(len(ledger), 2)
        self.assertEqual(set(ledger["schwab_symbol"]), {"/RBN26", "/HON26"})
        self.assertTrue((ledger["provenance_status"] == "verified").all())


if __name__ == "__main__":
    unittest.main()
