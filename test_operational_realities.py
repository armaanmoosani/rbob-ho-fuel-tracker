import unittest
import threading
import json
import os
import tempfile
from unittest.mock import patch

import pandas as pd

import weekly_report


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


if __name__ == "__main__":
    unittest.main()
