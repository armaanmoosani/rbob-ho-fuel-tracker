import unittest
import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
import validate_data
from calibration_artifacts import artifact_id, canonical_json

class TestValidateData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "graves_history.csv")
        self.log_path = os.path.join(self.temp_dir, "prediction_log.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _calibration_artifact(self, effective_session="2026-05-20"):
        artifact = {
            "artifact_schema_version": 2,
            "effective_session": effective_session,
            "training_start": "2026-01-01",
            "training_end": "2026-05-19",
            "purge_rows": 1,
            "source_history_hash": "d" * 64,
            "source_row_count": 100,
            "candidate_grid_version": "test-v2",
            "objective": "test objective",
            "prior_artifact_id": "bootstrap_config",
            "calibration": {
                "RB_nymex_daily_std": 8.0,
                "RB_HIKE_THRESHOLD_CENTS": 1.8,
                "RB_DROP_THRESHOLD_CENTS": -1.5,
                "RB_LEAN_HIKE_CENTS": 0.9,
                "RB_LEAN_DROP_CENTS": -0.8,
                "RB_window_days": 240,
                "LAG_DAYS": 0,
            },
            "candidate_grid": {"window": 240},
            "smoothing_input": {"window": 240},
            "generated_at": "2026-05-19T21:00:00-05:00",
        }
        artifact["artifact_id"] = artifact_id(artifact)
        return artifact

    def _captured_prediction(self, calibration_ref):
        return {
            "timestamp": "2026-05-20T14:35:00-05:00",
            "commodity": "RB",
            "predicted_direction": "HIKE",
            "nymex_move_cents": 2.0,
            "lag_used": 0,
            "window_used": 240,
            "threshold_used": 1.8,
            "actual_next_day_move_cents": "PENDING",
            "prediction_source": "live",
            "signal_contract": "/RBM26",
            "baseline_contract": "/RBM26",
            "settlement_source": "schwab",
            "baseline_source": "schwab_close_price",
            "settlement_captured_at": "2026-05-20T13:30:00-05:00",
            "contract_provenance_status": "verified",
            "log_schema_version": "3",
            "nymex_daily_std_used": 8.0,
            "z_score_used": 0.25,
            "conviction_label": "Moderate confidence (80%) | p=0.8000",
            "conviction_provenance": "passthrough_model_v2",
            "hike_threshold_used": 1.8,
            "drop_threshold_used": -1.5,
            "lean_hike_threshold_used": 0.9,
            "lean_drop_threshold_used": -0.8,
            "signal_price_used": 2.02,
            "baseline_price_used": 2.0,
            "runtime_config_hash": "a" * 64,
            "config_file_hash": "b" * 64,
            "metrics_cache_hash": "c" * 64,
            "calibration_effective_session": "2026-05-20",
            "calibration_artifact_id": calibration_ref,
        }

    def _write_artifact(self, artifact):
        path = os.path.join(self.temp_dir, "calibration_runs.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(artifact) + "\n")
        return path

    def test_is_cme_holiday(self):
        # Good Friday 2026 was April 3rd
        self.assertTrue(validate_data.is_cme_holiday(date(2026, 4, 3)))
        # Christmas 2026 was December 25th
        self.assertTrue(validate_data.is_cme_holiday(date(2026, 12, 25)))
        # A normal business day (e.g. Wednesday May 20, 2026)
        self.assertFalse(validate_data.is_cme_holiday(date(2026, 5, 20)))

    def test_repair_csv_if_corrupted_malformed_row(self):
        # Malformed last line should be pruned
        with open(self.csv_path, "w") as f:
            f.write("date,nymex_rb,nymex_ho,rack_u,rack_p,rack_d\n")
            f.write("2026-05-20,2.10,2.20,2.30,2.40,2.50\n")
            f.write("2026-05-21,2.15,2.25,2.35\n") # truncated
        validate_data.repair_csv_if_corrupted(self.csv_path)
        df = pd.read_csv(self.csv_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "nymex_rb"], 2.10)

    def test_repair_csv_if_corrupted_clean(self):
        # A clean file should remain unchanged
        df_orig = pd.DataFrame({
            "date": ["2026-05-20"],
            "nymex_rb": [2.10],
            "nymex_ho": [2.20],
            "rack_u": [2.30],
            "rack_p": [2.40],
            "rack_d": [2.50]
        })
        df_orig.to_csv(self.csv_path, index=False)
        validate_data.repair_csv_if_corrupted(self.csv_path)
        df_new = pd.read_csv(self.csv_path)
        self.assertEqual(len(df_new), 1)
        self.assertEqual(df_new.loc[0, "nymex_rb"], 2.10)

    def test_validate_prediction_log_missing_columns(self):
        # Missing column actual_next_day_move_cents should trigger sys.exit(1)
        df = pd.DataFrame({
            "timestamp": ["2026-05-20T12:00:00-05:00"],
            "commodity": ["RB"],
            "predicted_direction": ["HIKE"]
        })
        df.to_csv(self.log_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path)

    def test_validate_prediction_log_invalid_commodity(self):
        df = pd.DataFrame({
            "timestamp": ["2026-05-20T12:00:00-05:00"],
            "commodity": ["XYZ"], # invalid
            "predicted_direction": ["HIKE"],
            "actual_next_day_move_cents": ["PENDING"]
        })
        df.to_csv(self.log_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path)

    def test_validate_prediction_log_invalid_direction(self):
        df = pd.DataFrame({
            "timestamp": ["2026-05-20T12:00:00-05:00"],
            "commodity": ["RB"],
            "predicted_direction": ["UP"], # invalid, should be HIKE
            "actual_next_day_move_cents": ["PENDING"]
        })
        df.to_csv(self.log_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path)

    def test_validate_prediction_log_invalid_move(self):
        df = pd.DataFrame({
            "timestamp": ["2026-05-20T12:00:00-05:00"],
            "commodity": ["RB"],
            "predicted_direction": ["HIKE"],
            "actual_next_day_move_cents": ["INVALID_FLOAT"] # invalid
        })
        df.to_csv(self.log_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path)

    def test_validate_prediction_log_valid(self):
        df = pd.DataFrame({
            "timestamp": ["2026-05-20T12:00:00-05:00", "2026-05-21T12:00:00-05:00"],
            "commodity": ["RB", "HO"],
            "predicted_direction": ["HIKE", "DROP"],
            "actual_next_day_move_cents": ["PENDING", "1.5"],
            "prediction_source": ["live", "backfill"],
            "signal_contract": ["/RBM26", "unknown"],
            "baseline_contract": ["/RBM26", "unknown"],
            "settlement_source": ["schwab", "historical_import"],
            "baseline_source": ["schwab_close_price", "graves_history"],
            "settlement_captured_at": ["2026-05-20T13:35:00-05:00", ""],
            "contract_provenance_status": ["verified", "unknown"],
            "log_schema_version": ["2", "2"],
            "nymex_daily_std_used": ["unknown", "unknown"],
            "z_score_used": ["unknown", "unknown"],
            "conviction_label": ["unknown", "unknown"],
            "conviction_provenance": ["unknown", "unknown"],
            "hike_threshold_used": ["unknown", "unknown"],
            "drop_threshold_used": ["unknown", "unknown"],
            "lean_hike_threshold_used": ["unknown", "unknown"],
            "lean_drop_threshold_used": ["unknown", "unknown"],
            "signal_price_used": ["unknown", "unknown"],
            "baseline_price_used": ["unknown", "unknown"],
            "runtime_config_hash": ["unknown", "unknown"],
            "config_file_hash": ["unknown", "unknown"],
            "metrics_cache_hash": ["unknown", "unknown"],
            "calibration_effective_session": ["unknown", "unknown"],
            "calibration_artifact_id": ["unknown", "unknown"],
        })
        df.to_csv(self.log_path, index=False)
        # Should not raise exception
        validate_data.validate_prediction_log(self.log_path)

    def test_validate_prediction_log_accepts_immutable_calibration_artifact(self):
        artifact = self._calibration_artifact()
        artifact_path = self._write_artifact(artifact)
        pd.DataFrame([self._captured_prediction(artifact["artifact_id"])]).to_csv(
            self.log_path, index=False)

        validate_data.validate_prediction_log(self.log_path, artifact_path)

    def test_validate_prediction_log_accepts_legacy_metrics_reference(self):
        row = self._captured_prediction("metrics:" + "c" * 16)
        pd.DataFrame([row]).to_csv(self.log_path, index=False)

        validate_data.validate_prediction_log(self.log_path)

    def test_validate_prediction_log_rejects_unknown_immutable_artifact(self):
        artifact = self._calibration_artifact()
        artifact_path = self._write_artifact(artifact)
        pd.DataFrame([self._captured_prediction("f" * 64)]).to_csv(
            self.log_path, index=False)

        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path, artifact_path)

    def test_validate_prediction_log_rejects_artifact_session_mismatch(self):
        artifact = self._calibration_artifact(effective_session="2026-05-21")
        artifact_path = self._write_artifact(artifact)
        pd.DataFrame([self._captured_prediction(artifact["artifact_id"])]).to_csv(
            self.log_path, index=False)

        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path, artifact_path)

    def test_validate_prediction_log_rejects_artifact_calibration_mismatch(self):
        artifact = self._calibration_artifact()
        artifact_path = self._write_artifact(artifact)
        row = self._captured_prediction(artifact["artifact_id"])
        row["hike_threshold_used"] = 1.81
        row["threshold_used"] = 1.81
        pd.DataFrame([row]).to_csv(self.log_path, index=False)

        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path, artifact_path)

    def test_validate_prediction_log_invalid_source(self):
        df = pd.DataFrame({
            "timestamp": ["2026-05-20T12:00:00-05:00"],
            "commodity": ["RB"],
            "predicted_direction": ["HIKE"],
            "actual_next_day_move_cents": ["PENDING"],
            "prediction_source": ["estimated"]
        })
        df.to_csv(self.log_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_prediction_log(self.log_path)

    def test_validate_graves_history_negative_price(self):
        df = pd.DataFrame({
            "date": ["2026-05-20"],
            "nymex_rb": [-2.10], # negative!
            "nymex_ho": [2.20],
            "rack_u": [2.30],
            "rack_p": [2.40],
            "rack_d": [2.50]
        })
        df.to_csv(self.csv_path, index=False)
        with self.assertRaises(SystemExit):
            validate_data.validate_graves_history(self.csv_path)

    def test_validate_graves_history_valid(self):
        df = pd.DataFrame({
            "date": ["2026-05-20"],
            "nymex_rb": [2.10],
            "nymex_ho": [2.20],
            "rack_u": [2.30],
            "rack_p": [2.40],
            "rack_d": [2.50]
        })
        df.to_csv(self.csv_path, index=False)
        validate_data.validate_graves_history(self.csv_path)

    def test_validate_and_update_hashes(self):
        # Create graves_history.csv and config.json
        df = pd.DataFrame({
            "date": ["2026-05-20"],
            "nymex_rb": [2.10],
            "nymex_ho": [2.20],
            "rack_u": [2.30],
            "rack_p": [2.40],
            "rack_d": [2.50]
        })
        df.to_csv(self.csv_path, index=False)
        
        config_path = os.path.join(self.temp_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"test": true}')
            
        validate_data.validate_and_update_hashes(self.temp_dir)
        hash_file = os.path.join(self.temp_dir, "integrity_hashes.csv")
        self.assertTrue(os.path.exists(hash_file))
        
        df_hashes = pd.read_csv(hash_file)
        self.assertTrue(len(df_hashes) >= 2)
        
        # Verify running again with unmodified files succeeds
        validate_data.validate_and_update_hashes(self.temp_dir)

if __name__ == "__main__":
    unittest.main()
