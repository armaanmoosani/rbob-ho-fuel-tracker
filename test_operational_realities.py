import unittest
import threading
import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import weekly_report
import ingest_prices
import alignment
import model
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
    def test_main_and_replay_prefer_metrics_cache_over_config(self):
        """The live cache must win over the static config, everywhere.

        verify_statistics is deliberately absent: it validates the model from
        the history rather than from cached thresholds, so it has no config
        overlay to get wrong.
        """
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

        self.assertEqual(main_cfg["RB_HIKE_THRESHOLD_CENTS"], 2.99)
        self.assertEqual(replay_cfg["RB_HIKE_THRESHOLD_CENTS"], 2.99)
        self.assertFalse(hasattr(verify_statistics, "load_config"),
                         "verify_statistics must not read cached thresholds")


class TestContractProvenance(unittest.TestCase):
    def test_graves_history_is_never_used_as_a_crude_oil_baseline(self):
        import main

        self.assertEqual(main.graves_nymex_column_index("RB"), 1)
        self.assertEqual(main.graves_nymex_column_index("HO"), 2)
        self.assertIsNone(main.graves_nymex_column_index("CL"))

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


def _fitted_cache(prefix, slope=0.7, sigma=3.0, seed=31):
    """A realistic serialised pass-through model for live-signal tests.

    build_rack_signal suppresses the verdict entirely when no model is cached,
    so every live-path test must supply one.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 8, 400)
    y = slope * x + rng.normal(0, sigma, 400)
    return model.passthrough_to_config(model.fit_passthrough(x, y), prefix)


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
        runtime_config.update(_fitted_cache("RB"))
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

        self.assertIn("confidence", signal["conviction"])
        self.assertEqual(str(log.loc[0, "log_schema_version"]), "3")
        self.assertEqual(log.loc[0, "conviction_provenance"], "passthrough_model_v2")
        # The label carries the probability itself, so the claim can be
        # re-derived from the log rather than trusted.
        self.assertRegex(log.loc[0, "conviction_label"],
                         r"^(Low|Moderate|High) confidence \(\d+%\) \| p=0\.\d+$")
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

        # The band is read from the label recorded at decision time, so a later
        # change to the runtime config cannot retroactively re-grade the alert.
        recorded = log.loc[0, "conviction_label"].split(" |")[0]
        band = recorded.split(" confidence")[0] + " confidence"
        self.assertEqual(summary[band]["alerts"], 1)
        self.assertEqual(summary[band]["precision"], 100.0)
        self.assertEqual(sum(v["alerts"] for k, v in summary.items() if k != band), 0)

    def test_lean_signal_logs_the_threshold_that_actually_triggered(self):
        import main

        runtime_config = {
            "RB_HIKE_THRESHOLD_CENTS": 1.0,
            "RB_DROP_THRESHOLD_CENTS": -1.0,
            "RB_LEAN_HIKE_CENTS": 0.5,
            "RB_LEAN_DROP_CENTS": -0.5,
            "RB_nymex_daily_std": 10.0,
        }
        runtime_config.update(_fitted_cache("RB"))
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
            log["z_score_used"] = log["z_score_used"].astype(str)
            log.loc[0, "z_score_used"] = "0.1"
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
    def _history(rows=420, seed=17):
        """Synthetic history inside the alignment-verified era.

        Dates must start on or after alignment.CALIBRATION_ERA_START, because
        calibration now refuses rows whose NYMEX/rack pairing has not been
        verified.  The series carries a genuine positive pass-through so the
        model can be fitted.
        """
        start = pd.Timestamp(alignment.CALIBRATION_ERA_START)
        dates = pd.date_range(start, periods=rows, freq="B")
        rng = np.random.default_rng(seed)
        nymex_step = rng.normal(0, 0.03, rows)
        rb = 2.10 + nymex_step.cumsum()
        ho = 2.30 + (nymex_step * 1.1).cumsum()
        rack_u = rb.copy()
        rack_d = ho.copy()
        # rack follows the settle with ~0.7 pass-through plus idiosyncratic noise
        rack_u[1:] = rack_u[0] + (0.7 * nymex_step[1:] + rng.normal(0, 0.01, rows - 1)).cumsum()
        rack_d[1:] = rack_d[0] + (0.9 * nymex_step[1:] * 1.1 + rng.normal(0, 0.01, rows - 1)).cumsum()
        return pd.DataFrame({
            "date": dates, "nymex_rb": rb, "nymex_ho": ho,
            "rack_u": rack_u, "rack_p": rb + 0.10, "rack_d": rack_d,
        })

    @staticmethod
    def _cfg():
        return dict(backtest.DEFAULTS)

    def test_walk_forward_blocks_are_ordered_and_purged(self):
        """Train, purge and test must never overlap, in any fold."""
        frame = pd.DataFrame({
            "date": pd.bdate_range("2025-08-01", periods=400),
            "delta_nymex": np.linspace(-10, 10, 400),
            "delta_rack": np.linspace(-7, 7, 400),
        })
        result = backtest.walk_forward_evaluation(frame, self._cfg())
        self.assertEqual(len(result["folds"]), backtest.EVAL_FOLDS)

        n = len(frame)
        spans = []
        for fold_index in range(backtest.EVAL_FOLDS):
            test_end = n - fold_index * backtest.EVAL_TEST_ROWS
            test_start = test_end - backtest.EVAL_TEST_ROWS
            train_end = test_start - backtest.CALIBRATION_PURGE_ROWS
            train_start = max(0, train_end - self._cfg()["ROLLING_WINDOW_DAYS"])
            train = set(range(train_start, train_end))
            purge = set(range(train_end, test_start))
            test = set(range(test_start, test_end))
            self.assertTrue(train.isdisjoint(purge))
            self.assertTrue(train.isdisjoint(test))
            self.assertTrue(purge.isdisjoint(test))
            self.assertEqual(len(purge), backtest.CALIBRATION_PURGE_ROWS)
            spans.append((test_start, test_end))
        # Test blocks tile the recent history without overlapping each other.
        for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
            self.assertEqual(b_end, a_start)

    def test_live_calibration_is_idempotent_for_identical_history(self):
        source_hash = "c" * 64
        current = {
            "CALIBRATION_EFFECTIVE_SESSION": "2026-07-21",
            "CALIBRATION_SOURCE_HISTORY_HASH": source_hash,
            "CALIBRATION_METHOD_VERSION": backtest.CALIBRATION_METHOD_VERSION,
        }
        self.assertTrue(backtest.calibration_is_current(
            current, source_hash, "2026-07-21", "2026-07-20"))
        self.assertFalse(backtest.calibration_is_current(
            current, "d" * 64, "2026-07-21", "2026-07-20"))
        self.assertFalse(backtest.calibration_is_current(
            current, source_hash, "2026-07-22", "2026-07-21"))

    def test_method_change_forces_recalibration(self):
        """A cache from a superseded engine must never be served as current."""
        stale = {
            "CALIBRATION_EFFECTIVE_SESSION": "2026-07-21",
            "CALIBRATION_SOURCE_HISTORY_HASH": "c" * 64,
            "CALIBRATION_METHOD_VERSION": "v1-purged-three-fold",
        }
        self.assertFalse(backtest.calibration_is_current(
            stale, "c" * 64, "2026-07-21", "2026-07-20"))

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
        self.assertEqual(saved["CALIBRATION_METHOD_VERSION"],
                         backtest.CALIBRATION_METHOD_VERSION)

    def test_metrics_cache_drops_superseded_keys(self):
        """Stale in-sample statistics must not survive in the live cache."""
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            backtest, "METRICS_CACHE_PATH", os.path.join(temp_dir, "metrics.json")
        ):
            backtest.save_metrics_cache({
                "RB_HIKE_THRESHOLD_CENTS": 1.85,
                "RB_historical_win_rate": 0.9436,
                "RB_average_savings": 5.1889,
                "RB_high_z_win_rate": 0.83,
                "RB_opt_Hp": 20,
            }, effective_session="2026-07-21", source_history_hash="a" * 64)
            with open(backtest.METRICS_CACHE_PATH, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertIn("RB_HIKE_THRESHOLD_CENTS", saved)
        for dropped in ("RB_historical_win_rate", "RB_average_savings",
                        "RB_high_z_win_rate", "RB_opt_Hp"):
            self.assertNotIn(dropped, saved)

    def test_calibration_effective_session_skips_holidays_and_weekends(self):
        self.assertEqual(backtest._next_nymex_business_session("2026-07-02"), "2026-07-06")
        self.assertEqual(backtest._next_nymex_business_session("2026-07-17"), "2026-07-20")

    def test_future_rows_cannot_change_artifact_training_or_calibration(self):
        history = self._history()
        effective_session = history.loc[400, "date"].date().isoformat()
        first = backtest.build_shadow_calibration_artifact(
            history, self._cfg(), effective_session=effective_session)
        mutated = history.copy()
        # These rows are after the decision session and must not influence it.
        mutated.loc[401:, ["nymex_rb", "nymex_ho", "rack_u", "rack_d"]] = 9.99
        second = backtest.build_shadow_calibration_artifact(
            mutated, self._cfg(), effective_session=effective_session)
        self.assertEqual(first["training_end"], second["training_end"])
        self.assertEqual(first["source_history_hash"], second["source_history_hash"])
        self.assertEqual(first["calibration"], second["calibration"])
        self.assertEqual(first["artifact_id"], second["artifact_id"])

    def test_artifact_calibration_does_not_depend_on_the_live_cache(self):
        """The artifact must be a function of history and policy only.

        The engine no longer blends the previous threshold into the new one, so
        a corrupted live cache cannot leak into the point-in-time record.
        """
        history = self._history()
        session = history.loc[400, "date"].date().isoformat()
        baseline = backtest.build_shadow_calibration_artifact(
            history, self._cfg(), effective_session=session)

        polluted = self._cfg()
        polluted["RB_HIKE_THRESHOLD_CENTS"] = 99.0
        polluted["HO_DROP_THRESHOLD_CENTS"] = -99.0
        from_polluted = backtest.build_shadow_calibration_artifact(
            history, polluted, effective_session=session)

        self.assertEqual(baseline["calibration"], from_polluted["calibration"])

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
                pd.DataFrame(), "2026-07-21", artifact_path=path)
            self.assertEqual(replayed, written["calibration"])
            conflicting = dict(artifact)
            conflicting["calibration"] = {"RB_HIKE_THRESHOLD_CENTS": 1.80}
            with self.assertRaises(ValueError):
                append_calibration_artifact(path, conflicting)
            with self.assertRaises(CalibrationArtifactUnavailable):
                replay_day.simulate_thresholds_at_date(
                    pd.DataFrame(), "2026-07-22", artifact_path=path)

    def test_same_session_shadow_rerun_uses_the_original_prior_state(self):
        history = self._history()
        session = backtest._next_nymex_business_session(history["date"].iloc[-1])
        _, eligible = backtest._eligible_training_history(history, session)
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            backtest, "CALIBRATION_RUNS_PATH", os.path.join(temp_dir, "runs.jsonl")
        ), patch.object(backtest, "build_shadow_calibration_artifact") as build:
            artifact = {
                "artifact_schema_version": 1,
                "effective_session": session,
                "training_start": eligible["date"].iloc[0].date().isoformat(),
                "training_end": eligible["date"].iloc[-1].date().isoformat(),
                "purge_rows": 1,
                "source_history_hash": backtest._history_hash(eligible),
                "source_row_count": len(eligible),
                "candidate_grid_version": "test",
                "candidate_grid": {"windows": [90]},
                "objective": "test",
                "smoothing_input": {"BLEND_ALPHA": 0.3},
                "prior_artifact_id": "bootstrap_config",
                "calibration": {"RB_HIKE_THRESHOLD_CENTS": 1.23},
                "generated_at": "2026-07-20T18:00:00-05:00",
            }
            append_calibration_artifact(backtest.CALIBRATION_RUNS_PATH, artifact)
            written, created = backtest.write_shadow_calibration_artifact(
                history, self._cfg())
            self.assertFalse(created)
            self.assertEqual(written["calibration"], artifact["calibration"])
            build.assert_not_called()

    def test_calibration_refuses_history_outside_the_verified_era(self):
        """Legacy rows must not be able to re-enter calibration."""
        legacy = self._history()
        legacy["date"] = pd.date_range("2023-03-07", periods=len(legacy), freq="B")
        with self.assertRaises(ValueError):
            backtest._eligible_training_history(legacy, None)

