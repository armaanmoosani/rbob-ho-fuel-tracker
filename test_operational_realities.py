import unittest
import threading
import json
import os
import tempfile
from unittest.mock import patch

import pandas as pd

import weekly_report
import ingest_prices
import validate_data
import backtest
import replay_day
from calibration_artifacts import (
    CalibrationArtifactUnavailable,
    append_calibration_artifact,
)


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

    def test_tracker_commits_prediction_before_rebase(self):
        workflow_path = os.path.join(
            os.path.dirname(__file__), ".github", "workflows", "tracker.yml"
        )
        with open(workflow_path, "r", encoding="utf-8") as handle:
            workflow = handle.read()
        persist_step = workflow.split(
            "- name: Persist live prediction and settlement provenance", 1
        )[1]
        self.assertLess(
            persist_step.index('git commit -m "Record live fuel provenance"'),
            persist_step.index("git pull --rebase origin main"),
        )
        self.assertIn("git add data/prediction_log.csv", persist_step)
        self.assertIn("[ -f data/daily_settlement.json ]", persist_step)

    def test_policy_savings_formula_is_identical_for_hike_drop_and_flat(self):
        savings = weekly_report.policy_savings_cents(
            ["HIKE", "DROP", "FLAT"], [3.0, -4.0, 100.0]
        )
        self.assertEqual(savings, 7.0)

    def test_permutation_keeps_rb_and_ho_outcomes_in_date_blocks(self):
        rows = []
        for day in range(35):
            timestamp = pd.Timestamp("2026-01-02", tz="America/Chicago") + pd.Timedelta(days=day)
            rows.extend([
                {
                    "timestamp_dt": timestamp,
                    "commodity": "RB",
                    "predicted_direction": "HIKE" if day % 2 else "DROP",
                    "actual_move": float(day),
                },
                {
                    "timestamp_dt": timestamp,
                    "commodity": "HO",
                    "predicted_direction": "HIKE" if day % 3 else "DROP",
                    "actual_move": float(100 + day),
                },
            ])
        frame = pd.DataFrame(rows)
        original_policy = weekly_report.policy_savings_cents
        observed_move_vectors = []

        def capture_policy(directions, moves):
            observed_move_vectors.append(pd.Series(moves, dtype=float).to_numpy())
            return original_policy(directions, moves)

        with patch("weekly_report.policy_savings_cents", side_effect=capture_policy):
            p_value, real = weekly_report.date_block_permutation_test(
                frame, n_perm=200, seed=42
            )
        self.assertGreaterEqual(p_value, 1 / 201)
        self.assertLessEqual(p_value, 1.0)
        self.assertEqual(real, weekly_report.policy_savings_cents(
            frame["predicted_direction"], frame["actual_move"]
        ))
        for permuted in observed_move_vectors[1:]:
            self.assertTrue(((permuted[1::2] - permuted[0::2]) == 100.0).all())

    def test_permutation_rejects_duplicate_session_commodity_rows(self):
        timestamp = pd.Timestamp("2026-07-20", tz="America/Chicago")
        duplicate = pd.DataFrame({
            "timestamp_dt": [timestamp, timestamp],
            "commodity": ["RB", "RB"],
            "predicted_direction": ["HIKE", "DROP"],
            "actual_move": [1.0, -1.0],
        })
        with self.assertRaisesRegex(ValueError, "duplicate commodity/session"):
            weekly_report.date_block_permutation_test(duplicate, n_perm=10)


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
        self.assertEqual(log.loc[0, "conviction_provenance"], "suppressed")
        self.assertEqual(log.loc[0, "conviction_label"], "Not evaluated")

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

    def test_daily_settlement_v2_requires_matching_contract_identity(self):
        settlement = {
            "settlement_schema_version": 2,
            "date": "2026-07-20",
            "captured_at": "2026-07-20T13:35:00-05:00",
            "rbob_settlement": 2.20,
            "rbob_contract": "/RBN26",
            "heating_oil_settlement": 2.30,
            "heating_oil_contract": "/HON26",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "daily_settlement.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(settlement, handle)
            validate_data.validate_daily_settlement(path)
            settlement["rbob_contract"] = "/HON26"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(settlement, handle)
            with self.assertRaises(SystemExit):
                validate_data.validate_daily_settlement(path)


class TestConvictionProvenance(unittest.TestCase):
    def _build_live_signal(self, temp_dir):
        import main

        runtime_config = {
            "LAG_DAYS": 0,
            "ROLLING_WINDOW_DAYS": 120,
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "RB_nymex_daily_std": 10.0,
        }
        data = {
            "current_price": 2.10,
            "yesterday_close": 2.00,
            "schwab_symbol": "/RBN26",
            "baseline_schwab_symbol": "/RBN26",
            "baseline_source": "schwab_close_price",
            "data_source": "schwab",
            "contract_provenance_required": True,
        }
        now = pd.Timestamp("2026-07-20T14:35:00", tz="America/Chicago").to_pydatetime()
        with patch("main.DATA_DIR", temp_dir), patch("main.APP_CONFIG", runtime_config), patch(
            "main.load_settlement_snapshot", return_value=None
        ), patch("main.is_contract_roll_day", return_value=False), patch(
            "main.get_decoupling_warning", return_value=""
        ):
            signal = main.build_rack_signal("RB", data, now)
        return signal, runtime_config

    def test_live_prediction_captures_immutable_conviction_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            signal, _ = self._build_live_signal(temp_dir)
            log = pd.read_csv(os.path.join(temp_dir, "prediction_log.csv"))

        self.assertEqual(signal["conviction"], "Moderate Conviction")
        self.assertEqual(str(log.loc[0, "log_schema_version"]), "3")
        self.assertEqual(log.loc[0, "conviction_provenance"], "captured")
        self.assertEqual(log.loc[0, "conviction_label"], "Moderate Conviction")
        self.assertAlmostEqual(float(log.loc[0, "nymex_daily_std_used"]), 10.0)
        self.assertAlmostEqual(float(log.loc[0, "z_score_used"]), 1.0)
        self.assertRegex(log.loc[0, "runtime_config_hash"], r"^[0-9a-f]{64}$")

    def test_report_uses_stored_conviction_after_runtime_config_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, runtime_config = self._build_live_signal(temp_dir)
            log = pd.read_csv(os.path.join(temp_dir, "prediction_log.csv"))

        runtime_config["RB_nymex_daily_std"] = 1.0
        log["actual_move"] = [2.0]
        log["savings_cents"] = [2.0]
        log["is_correct"] = [True]
        summary = weekly_report.summarize_captured_convictions(log)

        self.assertEqual(summary["Moderate Conviction"]["alerts"], 1)
        self.assertEqual(summary["High Conviction"]["alerts"], 0)
        self.assertEqual(summary["Moderate Conviction"]["precision"], 100.0)

    def test_lean_signal_logs_the_threshold_that_actually_triggered(self):
        import main

        runtime_config = {
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "RB_nymex_daily_std": 10.0,
        }
        data = {
            "current_price": 2.006,
            "yesterday_close": 2.000,
            "schwab_symbol": "/RBN26",
            "baseline_schwab_symbol": "/RBN26",
            "baseline_source": "schwab_close_price",
            "data_source": "schwab",
            "contract_provenance_required": True,
        }
        now = pd.Timestamp("2026-07-20T14:35:00", tz="America/Chicago").to_pydatetime()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "main.DATA_DIR", temp_dir
        ), patch("main.APP_CONFIG", runtime_config), patch(
            "main.is_contract_roll_day", return_value=False
        ), patch(
            "main.load_settlement_snapshot", return_value=None
        ), patch(
            "main.get_decoupling_warning", return_value=""
        ):
            signal = main.build_rack_signal("RB", data, now)
            log_path = os.path.join(temp_dir, "prediction_log.csv")
            log = pd.read_csv(log_path)
            validate_data.validate_prediction_log(log_path)

        self.assertEqual(signal["action"], "LEAN_BUY")
        self.assertEqual(log.loc[0, "predicted_direction"], "HIKE")
        self.assertAlmostEqual(float(log.loc[0, "threshold_used"]), 0.5)

    def test_validator_rejects_inconsistent_live_z_score_and_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._build_live_signal(temp_dir)
            log_path = os.path.join(temp_dir, "prediction_log.csv")
            log = pd.read_csv(log_path)
            log.loc[0, "z_score_used"] = 0.1
            log.to_csv(log_path, index=False)
            with self.assertRaises(SystemExit):
                validate_data.validate_prediction_log(log_path)

    def test_unknown_legacy_conviction_is_excluded_and_new_live_unknown_is_rejected(self):
        legacy = pd.DataFrame({
            "predicted_direction": ["HIKE"],
            "conviction_provenance": ["unknown"],
            "conviction_label": ["High Conviction"],
            "is_correct": [True],
            "savings_cents": [3.0],
        })
        self.assertEqual(
            sum(item["alerts"] for item in weekly_report.summarize_captured_convictions(legacy).values()),
            0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _ = self._build_live_signal(temp_dir)
            log_path = os.path.join(temp_dir, "prediction_log.csv")
            log = pd.read_csv(log_path)
            log.loc[0, "conviction_provenance"] = "unknown"
            log.to_csv(log_path, index=False)
            with self.assertRaises(SystemExit):
                validate_data.validate_prediction_log(log_path)


class TestPointInTimeCalibrationArtifacts(unittest.TestCase):
    @staticmethod
    def _history(rows=500):
        dates = pd.date_range("2024-01-02", periods=rows, freq="B")
        movement = pd.Series(range(rows), dtype=float).mod(11).sub(5).to_numpy() / 1000
        rb = 2.10 + movement.cumsum()
        ho = 2.30 + (movement * 0.8).cumsum()
        return pd.DataFrame({
            "date": dates,
            "nymex_rb": rb,
            "nymex_ho": ho,
            "rack_u": rb + (movement * 0.2),
            "rack_p": rb + 0.10,
            "rack_d": ho + (movement * 0.2),
        })

    @staticmethod
    def _cfg():
        return {
            "BLEND_ALPHA": 0.3,
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "HO_HIKE_THRESHOLD_CENTS": 1.0,
            "HO_DROP_THRESHOLD_CENTS": -1.0,
            "CLAMP_HIKE_MIN": 0.3,
            "CLAMP_HIKE_MAX": 3.0,
            "CLAMP_DROP_MIN": -3.0,
            "CLAMP_DROP_MAX": -0.3,
        }

    @staticmethod
    def _small_grid():
        return {"windows": [90], "hike_percentiles": [15], "drop_percentiles": [85]}

    def test_purged_fold_boundaries_are_disjoint(self):
        folds = backtest.build_purged_walk_forward_folds(600, 120)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            train = set(range(fold["train_start"], fold["train_end"]))
            purge = set(range(fold["purge_start"], fold["purge_end"]))
            test = set(range(fold["test_start"], fold["test_end"]))
            self.assertTrue(train.isdisjoint(purge))
            self.assertTrue(train.isdisjoint(test))
            self.assertTrue(purge.isdisjoint(test))
            self.assertEqual(fold["purge_end"] - fold["purge_start"], 1)

    def test_live_calibration_is_idempotent_for_identical_history(self):
        source_hash = "c" * 64
        current = {
            "CALIBRATION_EFFECTIVE_SESSION": "2026-07-21",
            "CALIBRATION_SOURCE_HISTORY_HASH": source_hash,
        }
        self.assertTrue(backtest.calibration_is_current(
            current, source_hash, "2026-07-21", "2026-07-20"
        ))
        self.assertFalse(backtest.calibration_is_current(
            current, "d" * 64, "2026-07-21", "2026-07-20"
        ))
        self.assertFalse(backtest.calibration_is_current(
            current, source_hash, "2026-07-22", "2026-07-21"
        ))

    def test_legacy_cache_migrates_without_reapplying_smoothing(self):
        old_style = {"CALIBRATION_EFFECTIVE_SESSION": "2026-07-20"}
        self.assertTrue(backtest.calibration_is_current(
            old_style, "e" * 64, "2026-07-21", "2026-07-20"
        ))

    def test_metrics_cache_records_next_session_and_source_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            backtest, "METRICS_CACHE_PATH", os.path.join(temp_dir, "metrics.json")
        ):
            backtest.save_metrics_cache(
                {"RB_HIKE_THRESHOLD_CENTS": 2.99, "LAG_DAYS": 0},
                effective_session="2026-07-21",
                source_history_hash="f" * 64,
            )
            with open(backtest.METRICS_CACHE_PATH, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertEqual(saved["CALIBRATION_EFFECTIVE_SESSION"], "2026-07-21")
        self.assertEqual(saved["CALIBRATION_SOURCE_HISTORY_HASH"], "f" * 64)

    def test_calibration_effective_session_skips_holidays_and_weekends(self):
        self.assertEqual(
            backtest._next_nymex_business_session("2026-07-02"), "2026-07-06"
        )
        self.assertEqual(
            backtest._next_nymex_business_session("2026-07-17"), "2026-07-20"
        )

    def test_future_rows_cannot_change_artifact_training_or_calibration(self):
        history = self._history()
        kwargs = self._small_grid()
        effective_session = history.loc[450, "date"].date().isoformat()
        first = backtest.build_shadow_calibration_artifact(
            history, self._cfg(), effective_session=effective_session, **kwargs
        )
        mutated = history.copy()
        # These rows are after the decision session and must not influence it.
        mutated.loc[451:, ["nymex_rb", "nymex_ho", "rack_u", "rack_d"]] = 9.99
        second = backtest.build_shadow_calibration_artifact(
            mutated, self._cfg(), effective_session=effective_session, **kwargs
        )
        self.assertEqual(first["training_end"], second["training_end"])
        self.assertEqual(first["source_history_hash"], second["source_history_hash"])
        self.assertEqual(first["calibration"], second["calibration"])
        self.assertEqual(first["artifact_id"], second["artifact_id"])

    def test_artifacts_are_immutable_and_replay_ignores_current_cache(self):
        artifact = {
            "artifact_schema_version": 1,
            "effective_session": "2026-07-21",
            "training_start": "2025-01-01",
            "training_end": "2026-07-17",
            "purge_rows": 1,
            "source_history_hash": "a" * 64,
            "source_row_count": 300,
            "candidate_grid_version": "test",
            "candidate_grid": {"windows": [90]},
            "objective": "test",
            "smoothing_input": {"BLEND_ALPHA": 0.3},
            "prior_artifact_id": "bootstrap_config",
            "calibration": {"RB_HIKE_THRESHOLD_CENTS": 2.99, "LAG_DAYS": 0},
            "generated_at": "2026-07-20T18:00:00-05:00",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "calibration_runs.jsonl")
            written, created = append_calibration_artifact(path, artifact)
            self.assertTrue(created)
            replayed = replay_day.simulate_thresholds_at_date(
                pd.DataFrame(), "2026-07-21", artifact_path=path
            )
            self.assertEqual(replayed, written["calibration"])
            conflicting = dict(artifact)
            conflicting["calibration"] = {"RB_HIKE_THRESHOLD_CENTS": 1.80}
            with self.assertRaises(ValueError):
                append_calibration_artifact(path, conflicting)
            with self.assertRaises(CalibrationArtifactUnavailable):
                replay_day.simulate_thresholds_at_date(
                    pd.DataFrame(), "2026-07-22", artifact_path=path
                )

    def test_smoothing_uses_prior_artifact_not_current_cache(self):
        history = self._history()
        kwargs = self._small_grid()
        prior = backtest.build_shadow_calibration_artifact(
            history, self._cfg(), effective_session="2026-01-05", **kwargs
        )
        changed_cache = self._cfg()
        changed_cache["RB_HIKE_THRESHOLD_CENTS"] = 99.0
        changed_cache["HO_DROP_THRESHOLD_CENTS"] = -99.0
        from_prior = backtest.build_shadow_calibration_artifact(
            history, changed_cache, effective_session="2026-01-06",
            prior_artifact=prior, **kwargs
        )
        stable_base = self._cfg()
        expected = backtest.build_shadow_calibration_artifact(
            history, stable_base, effective_session="2026-01-06",
            prior_artifact=prior, **kwargs
        )
        self.assertEqual(from_prior["prior_artifact_id"], prior["artifact_id"])
        self.assertEqual(from_prior["calibration"], expected["calibration"])

    def test_same_session_shadow_rerun_uses_the_original_prior_state(self):
        history = self._history(rows=3)
        _, eligible = backtest._eligible_training_history(history, "2024-01-05")
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            backtest, "CALIBRATION_RUNS_PATH", os.path.join(temp_dir, "runs.jsonl")
        ), patch.object(backtest, "build_shadow_calibration_artifact") as build:
            artifact = {
                "artifact_schema_version": 1,
                "effective_session": "2024-01-05",
                "training_start": "2024-01-02",
                "training_end": "2024-01-04",
                "purge_rows": 1,
                "source_history_hash": backtest._history_hash(eligible),
                "source_row_count": 3,
                "candidate_grid_version": "test",
                "candidate_grid": {"windows": [90]},
                "objective": "test",
                "smoothing_input": {"BLEND_ALPHA": 0.3},
                "prior_artifact_id": "bootstrap_config",
                "calibration": {"RB_HIKE_THRESHOLD_CENTS": 2.99},
                "generated_at": "2026-07-20T18:00:00-05:00",
            }
            build.return_value = artifact
            _, created = backtest.write_shadow_calibration_artifact(history, self._cfg())
            self.assertTrue(created)
            _, created = backtest.write_shadow_calibration_artifact(history, self._cfg())
            self.assertFalse(created)
            self.assertEqual(build.call_args_list[0].kwargs["prior_artifact"], None)
            self.assertEqual(build.call_count, 1)

    def test_artifact_validator_rejects_broken_chain_and_impossible_dates(self):
        base = {
            "artifact_schema_version": 1,
            "effective_session": "2026-07-21",
            "training_start": "2026-01-02",
            "training_end": "2026-07-20",
            "purge_rows": 1,
            "source_history_hash": "a" * 64,
            "source_row_count": 100,
            "candidate_grid_version": "test",
            "candidate_grid": {"windows": [90]},
            "objective": "test",
            "smoothing_input": {"BLEND_ALPHA": 0.3},
            "prior_artifact_id": "bootstrap_config",
            "calibration": {"RB_HIKE_THRESHOLD_CENTS": 2.99},
            "generated_at": "2026-07-20T20:00:00-05:00",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "runs.jsonl")
            first, _ = append_calibration_artifact(path, base)
            broken = dict(base)
            broken.update({
                "effective_session": "2026-07-22",
                "prior_artifact_id": "b" * 64,
            })
            with self.assertRaisesRegex(ValueError, "continue the existing chain"):
                append_calibration_artifact(path, broken)
            impossible = dict(base)
            impossible["training_end"] = "2026-07-21"
            with self.assertRaisesRegex(ValueError, "impossible date boundaries"):
                append_calibration_artifact(os.path.join(temp_dir, "bad.jsonl"), impossible)
            self.assertRegex(first["artifact_id"], r"^[0-9a-f]{64}$")

    def test_schema_v1_artifact_remains_replayable_after_v2_upgrade(self):
        legacy = {
            "artifact_schema_version": 1,
            "effective_session": "2026-07-21",
            "training_start": "2026-01-02",
            "training_end": "2026-07-20",
            "purge_rows": 1,
            "source_history_hash": "a" * 64,
            "source_row_count": 100,
            "candidate_grid_version": "v1-purged-three-fold",
            "objective": "median_out_of_sample_savings_cents",
            "prior_artifact_id": "bootstrap_config",
            "calibration": {"RB_HIKE_THRESHOLD_CENTS": 2.99},
            "generated_at": "2026-07-20T20:00:00-05:00",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "runs.jsonl")
            written, _ = append_calibration_artifact(path, legacy)
            replayed = replay_day.simulate_thresholds_at_date(
                pd.DataFrame(), "2026-07-21", artifact_path=path
            )
        self.assertEqual(replayed, written["calibration"])


if __name__ == "__main__":
    unittest.main()
