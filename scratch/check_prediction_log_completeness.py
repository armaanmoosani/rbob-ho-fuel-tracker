import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Ensure parent directory is in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import backtest
import validate_data
from replay_day import simulate_thresholds_at_date

DATA_DIR = os.path.join(parent_dir, "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")

def main():
    parser = argparse.ArgumentParser(description="Prediction Log Completeness & Safe Backfiller")
    parser.add_argument("--backfill", action="store_true", help="Backfill missing predictions retroactively")
    args = parser.parse_args()
    
    if not os.path.exists(CSV_PATH):
        print(f"Graves history database missing at: {CSV_PATH}")
        sys.exit(1)
        
    df_hist = pd.read_csv(CSV_PATH)
    df_hist['parsed_date'] = pd.to_datetime(df_hist['date'])
    df_hist = df_hist.sort_values('date').reset_index(drop=True)
    
    # Load or initialize prediction log
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=[
            "timestamp", "commodity", "predicted_direction", "nymex_move_cents",
            "lag_used", "window_used", "threshold_used", "actual_next_day_move_cents",
            "prediction_source"
        ])
        
    # Standardize/ensure prediction_source exists
    if 'prediction_source' not in df_log.columns:
        df_log['prediction_source'] = 'unlabelled'
    df_log['prediction_source'] = df_log['prediction_source'].fillna('unlabelled')
    
    # Parse existing log dates for matching
    df_log['date_only'] = df_log['timestamp'].apply(lambda x: x.split('T')[0] if isinstance(x, str) else "")
    
    # Scan graves_history for candidate trading days (where NYMEX settlements exist)
    # The first row doesn't have a prior day, so we start from index 1
    candidate_rows = df_hist[
        (df_hist['nymex_rb'].notna()) & 
        (df_hist['nymex_ho'].notna())
    ].index.tolist()
    
    candidate_rows = [idx for idx in candidate_rows if idx > 0]
    
    print(f"Total candidate trading days in graves_history: {len(candidate_rows)}")
    
    missing_entries = []
    for idx in candidate_rows:
        row = df_hist.iloc[idx]
        target_date = row['date']
        
        # Check if RB exists in prediction log
        rb_exists = not df_log[
            (df_log['date_only'] == target_date) & 
            (df_log['commodity'] == 'RB')
        ].empty
        
        # Check if HO exists in prediction log
        ho_exists = not df_log[
            (df_log['date_only'] == target_date) & 
            (df_log['commodity'] == 'HO')
        ].empty
        
        if not rb_exists:
            missing_entries.append((idx, target_date, 'RB'))
        if not ho_exists:
            missing_entries.append((idx, target_date, 'HO'))
            
    print(f"Found {len(missing_entries)} missing prediction log entries.")
    
    if len(missing_entries) == 0:
        print("Prediction log is 100% complete and aligned with graves_history.csv!")
        sys.exit(0)
        
    if not args.backfill:
        print("Run with --backfill to safely reconstruct these missing historical entries.")
        sys.exit(0)
        
    print(f"\nStarting safe backfill for {len(missing_entries)} entries...")
    
    # Chronological sort of missing entries by index
    missing_entries = sorted(missing_entries, key=lambda x: x[0])
    
    # Cache for calibrated configurations at each date to avoid redundant walk-forward calibration
    config_cache = {}
    
    backfill_records = []
    skipped_count = 0
    
    for idx, target_date, comm in missing_entries:
        # Strict presence check assertion guard before processing/writing
        # reload or re-verify against current in-memory DataFrame
        in_memory_exists = not df_log[
            (df_log['date_only'] == target_date) & 
            (df_log['commodity'] == comm)
        ].empty
        
        if in_memory_exists:
            print(f"WARNING: Entry already exists in prediction log for {(target_date, comm)}. Skipping safely to prevent duplicate/overwrite.")
            skipped_count += 1
            continue
            
        # Get point-in-time calibrated configuration
        if target_date not in config_cache:
            calibrated_cfg = simulate_thresholds_at_date(df_hist, target_date)
            config_cache[target_date] = calibrated_cfg
        else:
            calibrated_cfg = config_cache[target_date]
            
        # Fetch actual nymex prices
        nymex_col = 'nymex_rb' if comm == 'RB' else 'nymex_ho'
        rack_col = 'rack_u' if comm == 'RB' else 'rack_d'
        
        curr_row = df_hist.iloc[idx]
        prev_row = df_hist.iloc[idx - 1]
        
        curr_nymex = curr_row[nymex_col]
        prev_nymex = prev_row[nymex_col]
        
        change_cents = (curr_nymex - prev_nymex) * 100
        
        hike_thresh = calibrated_cfg.get(f"{comm}_HIKE_THRESHOLD_CENTS", 1.0)
        drop_thresh = calibrated_cfg.get(f"{comm}_DROP_THRESHOLD_CENTS", -1.0)
        
        if change_cents >= hike_thresh:
            direction = "HIKE"
            thresh = hike_thresh
        elif change_cents <= drop_thresh:
            direction = "DROP"
            thresh = drop_thresh
        else:
            direction = "FLAT"
            thresh = 0.0
            
        # Calculate actual next day move in physical rack price
        curr_rack = curr_row[rack_col]
        prev_rack = prev_row[rack_col]
        
        if pd.notna(curr_rack) and pd.notna(prev_rack):
            actual_move = round((curr_rack - prev_rack) * 100, 2)
        else:
            actual_move = "PENDING"
            
        lag = calibrated_cfg.get("LAG_DAYS", 0)
        window = calibrated_cfg.get("ROLLING_WINDOW_DAYS", 120)
        
        # Build timestamp
        timestamp = f"{target_date}T14:35:00-05:00"
        
        new_record = {
            "timestamp": timestamp,
            "commodity": comm,
            "predicted_direction": direction,
            "nymex_move_cents": round(change_cents, 2),
            "lag_used": lag,
            "window_used": window,
            "threshold_used": round(thresh, 2),
            "actual_next_day_move_cents": actual_move,
            "prediction_source": "backfill"
        }
        
        backfill_records.append(new_record)
        
        # Add to the in-memory dataframe immediately to satisfy strict presence guard in subsequent iterations
        # (e.g. if we have multiple missing entries for the same day)
        new_row_df = pd.DataFrame([new_record])
        new_row_df['date_only'] = target_date
        df_log = pd.concat([df_log, new_row_df], ignore_index=True)
        
    print(f"\nConstructed {len(backfill_records)} backfill entries (Skipped: {skipped_count}).")
    
    if len(backfill_records) > 0:
        # Load existing CSV exactly as is to write safely
        existing_log_df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
        
        # Append new records
        df_to_append = pd.DataFrame(backfill_records)
        df_to_append = df_to_append[
            ["timestamp", "commodity", "predicted_direction", "nymex_move_cents",
             "lag_used", "window_used", "threshold_used", "actual_next_day_move_cents",
             "prediction_source"]
        ]
        
        if not existing_log_df.empty:
            # Ensure existing log has prediction_source
            if 'prediction_source' not in existing_log_df.columns:
                existing_log_df['prediction_source'] = 'unlabelled'
            existing_log_df['prediction_source'] = existing_log_df['prediction_source'].fillna('unlabelled')
            
            final_df = pd.concat([existing_log_df, df_to_append], ignore_index=True)
        else:
            final_df = df_to_append
            
        final_df.to_csv(LOG_PATH, index=False)
        print(f"Successfully appended {len(backfill_records)} backfill entries to prediction_log.csv!")
        
        # Cryptographically update data integrity hashes
        print("\nSyncing cryptographic integrity hashes...")
        validate_data.validate_all(DATA_DIR)
        print("Hashes successfully synced.")

if __name__ == "__main__":
    main()
