import os
import sys
import pandas as pd
import numpy as np

def main(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
    log_path = os.path.join(data_dir, "prediction_log.csv")
    if not os.path.exists(log_path):
        print(f"Prediction log not found at: {log_path}")
        return
        
    df = pd.read_csv(log_path)
    if df.empty:
        print("Prediction log is empty.")
        return
        
    # Standardize columns
    if 'prediction_source' not in df.columns:
        df['prediction_source'] = 'unlabelled'
    else:
        df['prediction_source'] = df['prediction_source'].fillna('unlabelled')
        
    # We want live predictions that have been resolved (not PENDING)
    df_live = df[
        (df['prediction_source'] == 'live') & 
        (df['actual_next_day_move_cents'] != 'PENDING')
    ].copy()
    
    if df_live.empty:
        print("No resolved live shadow-mode predictions found in prediction log.")
        return
        
    df_live['actual_move'] = pd.to_numeric(df_live['actual_next_day_move_cents'], errors='coerce')
    df_live = df_live.dropna(subset=['actual_move']).copy()
    
    print("======================================================================")
    print("                LIVE FORWARD TEST SHADOW-MODE VALIDATION              ")
    print("======================================================================")
    print(f"Total resolved live predictions found: {len(df_live)}")
    
    # Historical baseline precision ranges and floors
    ranges = {
        'RB': {'range': '53%–73%', 'floor': 0.53, 'name': 'RBOB Unleaded'},
        'HO': {'range': '60%–79%', 'floor': 0.60, 'name': 'Heating Oil Diesel'}
    }
    
    # We will bin alerts by conviction
    # Z-score conviction boundaries:
    # High: |Z| >= 1.5, Moderate: 1.0 <= |Z| < 1.5, Low: |Z| < 1.0
    # Wait, the threshold_used or nymex_move_cents can be used to calculate Z, or we can check the threshold_used vs nymex_daily_std?
    # Actually, in main.py, the conviction is based on Z = change_cents / nymex_daily_std
    # Wait, does prediction_log.csv store the Z-score or the nymex_daily_std?
    # Ah! The prediction_log.csv stores nymex_move_cents and threshold_used and nymex_daily_std is NOT explicitly in the columns,
    # but nymex_daily_std can be loaded from the config.json or we can approximate it, OR wait!
    # In prediction_log.csv:
    # timestamp,commodity,predicted_direction,nymex_move_cents,lag_used,window_used,threshold_used,actual_next_day_move_cents,prediction_source
    # Wait! Can we calculate the absolute Z-score of the prediction?
    # Yes! The Z-score is computed as change_cents / nymex_daily_std. The change_cents is nymex_move_cents.
    # In config.json, we have the calibrated nymex_daily_std!
    # Let's load nymex_daily_std from config.json to perform the Z-score calculation on the fly!
    
    config_path = os.path.join(data_dir, "config.json")
    daily_std = {'RB': 1.0, 'HO': 1.0}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            daily_std['RB'] = cfg.get("RB_nymex_daily_std", 1.0)
            daily_std['HO'] = cfg.get("HO_nymex_daily_std", 1.0)
        except Exception as e:
            print(f"Warning: Failed to load config.json for daily std: {e}")
            
    # Calculate Z-score for each prediction
    z_scores = []
    for idx, row in df_live.iterrows():
        comm = row['commodity']
        std = daily_std.get(comm, 1.0)
        move = row['nymex_move_cents']
        z = abs(move / std) if std > 0 else 0.0
        z_scores.append(z)
    df_live['abs_z'] = z_scores
    
    def get_conviction_label(abs_z):
        if abs_z >= 1.5:
            return "High (|Z| >= 1.5)"
        elif abs_z >= 1.0:
            return "Moderate (1.0 <= |Z| < 1.5)"
        else:
            return "Low (|Z| < 1.0)"
            
    df_live['conviction'] = df_live['abs_z'].apply(get_conviction_label)
    
    # Print table binned by Commodity and Conviction
    print("\nLive Precision metrics binned by conviction level:")
    print(f"{'Commodity':<10} | {'Conviction Level':<28} | {'Alerts':<6} | {'Precision':<24} | {'Status':<12}")
    print("-" * 90)
    
    all_validated = True
    
    for comm in ['RB', 'HO']:
        comm_df = df_live[df_live['commodity'] == comm]
        comm_info = ranges[comm]
        
        for conviction in ["High (|Z| >= 1.5)", "Moderate (1.0 <= |Z| < 1.5)", "Low (|Z| < 1.0)"]:
            bin_df = comm_df[comm_df['conviction'] == conviction]
            
            # Filter out FLAT directions as they are not active alerts
            active_df = bin_df[bin_df['predicted_direction'].isin(['HIKE', 'DROP'])].copy()
            count = len(active_df)
            
            if count < 20:
                precision_str = "insufficient live history"
                status = "N/A"
            else:
                # Calculate correct alerts
                correct = 0
                for idx, row in active_df.iterrows():
                    pred = row['predicted_direction']
                    actual = row['actual_move']
                    if pred == 'HIKE' and actual > 0:
                        correct += 1
                    elif pred == 'DROP' and actual < 0:
                        correct += 1
                
                prec = correct / count
                precision_str = f"{prec * 100:.1f}% ({correct}/{count})"
                
                # Check if within historical range or above the floor
                floor = comm_info['floor']
                if prec >= floor:
                    status = "VALIDATED"
                else:
                    status = "WARNING"
                    all_validated = False
                    
            print(f"{comm:<10} | {conviction:<28} | {count:<6} | {precision_str:<24} | {status:<12}")
            
    print("\nOverall Live Precision compared to Historical Range:")
    for comm in ['RB', 'HO']:
        comm_df = df_live[df_live['commodity'] == comm]
        active_df = comm_df[comm_df['predicted_direction'].isin(['HIKE', 'DROP'])].copy()
        count = len(active_df)
        comm_info = ranges[comm]
        
        if count == 0:
            print(f"- {comm_info['name']}: No active alerts fired.")
            continue
            
        correct = 0
        for idx, row in active_df.iterrows():
            pred = row['predicted_direction']
            actual = row['actual_move']
            if pred == 'HIKE' and actual > 0:
                correct += 1
            elif pred == 'DROP' and actual < 0:
                correct += 1
                
        prec = correct / count
        status = "PASSED" if prec >= comm_info['floor'] else "WARNING"
        print(f"- {comm_info['name']}: Live Precision = {prec*100:.1f}% ({correct}/{count}) vs Historical {comm_info['range']} | Status: {status}")
        
    print("======================================================================")
    
if __name__ == "__main__":
    import json
    main()
