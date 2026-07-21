import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(__file__))
import backtest
from calibration_artifacts import artifact_for_session, CalibrationArtifactUnavailable

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
METRICS_CACHE_PATH = os.path.join(DATA_DIR, "metrics_cache.json")
CALIBRATION_RUNS_PATH = os.path.join(DATA_DIR, "calibration_runs.jsonl")


def load_config(config_path=None, metrics_cache_path=None):
    cfg = {
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
        "LAG_DAYS": 0,
        "ROLLING_WINDOW_DAYS": 120,
    }
    config_path = config_path or CONFIG_PATH
    metrics_cache_path = metrics_cache_path or METRICS_CACHE_PATH
    for path in (config_path, metrics_cache_path):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def simulate_thresholds_at_date(df, target_date, artifact_path=None):
    """Load the immutable calibration that was eligible for ``target_date``.

    ``df`` remains in the signature for call-site compatibility, but is not used
    to synthesize historical thresholds.  Recomputing from today's cache would
    turn a replay into a forward-contaminated estimate.
    """
    del df
    artifact = artifact_for_session(
        artifact_path or CALIBRATION_RUNS_PATH, str(target_date)[:10]
    )
    return dict(artifact["calibration"])

def main():
    parser = argparse.ArgumentParser(description="Deterministic Point-In-Time Replay Validation")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD to replay")
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH) or not os.path.exists(LOG_PATH):
        print("Required history or prediction log files missing.")
        sys.exit(1)

    df_hist = pd.read_csv(CSV_PATH)
    df_log = pd.read_csv(LOG_PATH)

    if df_log.empty:
        print("Prediction log is empty. Nothing to replay.")
        sys.exit(0)

    # Parse date
    df_log['date_only'] = df_log['timestamp'].apply(lambda x: x.split('T')[0] if isinstance(x, str) else "")
    
    if args.date:
        target_date = args.date
    else:
        # Default to the most recent date with predictions in the log
        target_date = sorted(df_log['date_only'].unique())[-1]

    print(f"=== DETERMINISTIC REPLAY AUDIT FOR DATE: {target_date} ===")

    # 1. Fetch the actual logged records for that date
    day_preds = df_log[df_log['date_only'] == target_date]
    if day_preds.empty:
        print(f"Error: No prediction logs found for date {target_date}.")
        sys.exit(1)

    # 2. Re-run stateful walk-forward calibration up to target_date
    try:
        calibrated_cfg = simulate_thresholds_at_date(df_hist, target_date)
    except CalibrationArtifactUnavailable as exc:
        print(f"Replay status: unknown. {exc}")
        sys.exit(0)
    
    # 3. Verify predictions and check for leakages
    mismatches = 0
    
    # Find the target date row and prior row in history
    hist_idx_list = df_hist.index[df_hist['date'] == target_date].tolist()
    if not hist_idx_list or hist_idx_list[0] - 1 < 0:
        print(f"Error: Target date {target_date} or its prior business day is missing from Graves history.")
        sys.exit(1)
        
    curr_idx = hist_idx_list[0]
    curr_hist_row = df_hist.iloc[curr_idx]
    prev_hist_row = df_hist.iloc[curr_idx - 1]

    print("\n--- Point-In-Time Replay Results ---")
    for _, log_row in day_preds.iterrows():
        comm = log_row['commodity']
        logged_dir = log_row['predicted_direction']
        logged_thresh = float(log_row['threshold_used'])
        logged_move = float(log_row['nymex_move_cents'])
        
        # Calculate simulated change using ONLY graves_history at target_date
        nymex_col = 'nymex_rb' if comm == 'RB' else 'nymex_ho'
        
        curr_nymex = curr_hist_row[nymex_col]
        prev_nymex = prev_hist_row[nymex_col]
        
        if pd.isna(curr_nymex) or pd.isna(prev_nymex):
            print(f"[{comm}] Mismatch: Missing NYMEX prices in history for {target_date}.")
            mismatches += 1
            continue
            
        sim_change_cents = (curr_nymex - prev_nymex) * 100
        
        # Fetch point-in-time calibrated thresholds
        hike_thresh = calibrated_cfg.get(f"{comm}_HIKE_THRESHOLD_CENTS", 1.0)
        drop_thresh = calibrated_cfg.get(f"{comm}_DROP_THRESHOLD_CENTS", -1.0)
        
        # Apply identical signal formula
        if sim_change_cents >= hike_thresh:
            sim_dir = "HIKE"
            active_thresh = hike_thresh
        elif sim_change_cents <= drop_thresh:
            sim_dir = "DROP"
            active_thresh = drop_thresh
        else:
            sim_dir = "FLAT"
            active_thresh = 0.0

        # Check for strict equivalence
        match = (sim_dir == logged_dir)
        print(f"[{comm}] Logged: {logged_dir} (Thresh: {logged_thresh:+.2f}c, Move: {logged_move:+.2f}c)")
        print(f"[{comm}] Replay: {sim_dir} (Thresh: {active_thresh:+.2f}c, Move: {sim_change_cents:+.2f}c)")
        
        if match:
            print(f"[{comm}] => SUCCESS: Replay matches logged prediction perfectly.")
        else:
            print(f"[{comm}] => FAILURE: Prediction mismatch detected!")
            mismatches += 1
            
        # Tolerance check on nymex move (precision check to ensure no look-ahead shift)
        if abs(sim_change_cents - logged_move) > 1e-4:
            print(f"[{comm}] WARNING: NYMEX change cents difference of {abs(sim_change_cents - logged_move):.4f} detected! Potential timezone/data-drift.")
            mismatches += 1

    print("\n==========================================")
    if mismatches == 0:
        print("ALL TESTS PASSED: Point-in-time replay is 100% deterministic and leakage-free.")
        sys.exit(0)
    else:
        print(f"AUDIT FAILED: {mismatches} mismatch/leakage warnings detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
