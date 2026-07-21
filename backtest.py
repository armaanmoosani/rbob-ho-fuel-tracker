import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import validate_data
from futures_util import is_contract_roll_day
import pytz
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
METRICS_CACHE_PATH = os.path.join(DATA_DIR, "metrics_cache.json")

def load_config():
    defaults = {
        "MIN_ROWS_FOR_TUNING": 30,
        "PRICE_MIN": 1.50,
        "PRICE_MAX": 6.00,
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
        "CLAMP_HIKE_MIN": 0.3,
        "CLAMP_HIKE_MAX": 3.0,
        "CLAMP_DROP_MIN": -3.0,
        "CLAMP_DROP_MAX": -0.3
    }
    cfg = defaults.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                user_cfg = json.load(f)
                cfg.update(user_cfg)
        except Exception:
            pass
            
    # Overlay metrics_cache if it exists (Issue 1.1)
    if os.path.exists(METRICS_CACHE_PATH):
        try:
            with open(METRICS_CACHE_PATH, "r") as f:
                cache_cfg = json.load(f)
                cfg.update(cache_cfg)
        except Exception:
            pass
            
    return cfg

def save_metrics_cache(cfg):
    output_keys = [
        "ROLLING_WINDOW_DAYS",
        "LAG_DAYS"
    ]
    for k in cfg.keys():
        if k.startswith("RB_") or k.startswith("HO_"):
            output_keys.append(k)
    cache_data = {k: cfg[k] for k in output_keys if k in cfg}
    cache_data["CALIBRATION_EFFECTIVE_SESSION"] = datetime.now(
        pytz.timezone("America/Chicago")
    ).date().isoformat()
    
    tmp_path = METRICS_CACHE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    os.replace(tmp_path, METRICS_CACHE_PATH)


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def git_commit_push(message):
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", "data/metrics_cache.json", "data/integrity_hashes.csv"], check=True)
        # Check if there are changes before committing
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
            print("Successfully committed and pushed metrics changes.")
        else:
            print("No metrics changes to commit.")
    except Exception as e:
        print(f"Git commit/push failed: {e}")
        raise e

def get_clean_deltas(df, nymex_col, rack_col):
    """
    Cleans and computes the daily price changes in cents.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.get('date', pd.Series(dtype='object'))):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass

    # Drop weekends (dayofweek >= 5) first to avoid NaN diffs on Mondays (Issue 1.2)
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        df = df[df['date'].dt.dayofweek < 5].reset_index(drop=True)

    df['delta_nymex'] = df[nymex_col].diff() * 100
    df['delta_rack'] = df[rack_col].diff() * 100

    df_clean = df.dropna(subset=[nymex_col, rack_col, 'delta_nymex', 'delta_rack']).copy()
    return df_clean['delta_nymex'], df_clean['delta_rack']

def train_thresholds(delta_nymex, delta_rack, Hp, Dp, clamp_bounds=None):
    """
    Train thresholds on cleaned daily price changes.
    """
    if clamp_bounds is None:
        hike_min, hike_max, drop_min, drop_max = 0.3, 3.0, -3.0, -0.3
    else:
        hike_min, hike_max, drop_min, drop_max = clamp_bounds

    if len(delta_nymex) < 10:
        return 1.0, -1.0

    hike_mask = (delta_rack > 0) & (delta_nymex > 0)
    drop_mask = (delta_rack < 0) & (delta_nymex < 0)

    hike_thresh = 1.0
    if hike_mask.sum() >= 5:
        raw_hike = np.percentile(delta_nymex[hike_mask], Hp)
        hike_thresh = clamp(raw_hike, hike_min, hike_max)

    drop_thresh = -1.0
    if drop_mask.sum() >= 5:
        raw_drop = np.percentile(delta_nymex[drop_mask], Dp)
        drop_thresh = clamp(raw_drop, drop_min, drop_max)

    return hike_thresh, drop_thresh

def simulate_walk_forward(df, nymex_col, rack_col, W, Hp, Dp, prefix, clamp_bounds=None):
    """
    Simulates walk-forward out-of-sample testing on 3 folds.
    Returns the median savings across all folds.
    """
    # 3 folds, 90-day test window each
    folds = 3
    test_size = 90
    total_needed = test_size * folds
    if len(df) < total_needed + W:
        return -9999.0

    fold_savings = []

    for f in range(folds):
        # N = length of dataset
        N = len(df)
        # test window starts at N - (f+1)*90, ends at N - f*90
        test_start = N - (f + 1) * test_size
        test_end = N - f * test_size if f > 0 else N
        
        train_start = max(0, test_start - W)
        
        df_train = df.iloc[train_start:test_start]
        df_test = df.iloc[test_start:test_end]

        # Clean and get training deltas
        train_nymex, train_rack = get_clean_deltas(df_train, nymex_col, rack_col)
        hike_thresh, drop_thresh = train_thresholds(train_nymex, train_rack, Hp, Dp, clamp_bounds)

        # Evaluate on out-of-sample test window using pre-computed deltas to avoid boundary loss
        test_nymex = df_test['delta_nymex']
        test_rack = df_test['delta_rack']

        savings = 0.0
        for i in range(len(df_test)):
            dt = df_test['date'].iloc[i]
            ch = test_nymex.iloc[i]
            act = test_rack.iloc[i]
            if pd.isna(ch) or pd.isna(act):
                continue
            # Exclude contract roll days from OOS savings (Issue 8.1)
            if is_contract_roll_day(dt, prefix):
                continue
            if ch >= hike_thresh:
                savings += act
            elif ch <= drop_thresh:
                savings += -act
        
        fold_savings.append(savings)

    return np.median(fold_savings)

def run_optimization(df, nymex_col, rack_col, prefix, cfg):
    """
    Runs grid search over window and percentiles, finds the best parameters
    using median out-of-sample savings, trains final thresholds,
    and returns metrics.
    """
    # Expose clamp thresholds from config
    hike_min = cfg.get("CLAMP_HIKE_MIN", 0.3)
    hike_max = cfg.get("CLAMP_HIKE_MAX", 3.0)
    drop_min = cfg.get("CLAMP_DROP_MIN", -3.0)
    drop_max = cfg.get("CLAMP_DROP_MAX", -0.3)
    clamp_bounds = (hike_min, hike_max, drop_min, drop_max)

    # Pre-compute deltas on full history to prevent out-of-sample window boundary loss
    df = df.copy()
    df['delta_nymex'] = df[nymex_col].diff() * 100
    df['delta_rack'] = df[rack_col].diff() * 100

    windows = [90, 120, 180, 240]
    hike_percentiles = [10, 15, 20, 25]
    drop_percentiles = [75, 80, 85, 90]

    best_median_savings = -9999.0
    best_params = None

    # Sweep grid
    for W in windows:
        for Hp in hike_percentiles:
            for Dp in drop_percentiles:
                med_sav = simulate_walk_forward(df, nymex_col, rack_col, W, Hp, Dp, prefix, clamp_bounds)
                if med_sav > best_median_savings:
                    best_median_savings = med_sav
                    best_params = (W, Hp, Dp)
                elif med_sav == best_median_savings and best_params is not None:
                    # Tie breaker: choose larger window for stability
                    if W > best_params[0]:
                        best_params = (W, Hp, Dp)

    if best_params is None:
        best_params = (120, 15, 85) # default fallback

    opt_W, opt_Hp, opt_Dp = best_params
    print(f"[{prefix}] Optimal Parameters: Win={opt_W}, HikePct={opt_Hp}, DropPct={opt_Dp} | Med. OOS Savings={best_median_savings:+.2f}c")

    # Train final thresholds on the last opt_W days of history
    df_final_train = df.tail(opt_W)
    final_nymex, final_rack = get_clean_deltas(df_final_train, nymex_col, rack_col)
    raw_hike, raw_drop = train_thresholds(final_nymex, final_rack, opt_Hp, opt_Dp, clamp_bounds)

    # Smooth the thresholds against config
    old_hike = cfg.get(f"{prefix}_HIKE_THRESHOLD_CENTS", 1.0)
    old_drop = cfg.get(f"{prefix}_DROP_THRESHOLD_CENTS", -1.0)
    alpha = cfg.get("BLEND_ALPHA", 0.3)

    smoothed_hike = round(alpha * raw_hike + (1 - alpha) * old_hike, 2)
    smoothed_drop = round(alpha * raw_drop + (1 - alpha) * old_drop, 2)

    cfg[f"{prefix}_HIKE_THRESHOLD_CENTS"] = smoothed_hike
    cfg[f"{prefix}_DROP_THRESHOLD_CENTS"] = smoothed_drop

    # Calculate additional metrics on the final training set
    df_slice = df.tail(opt_W).copy()
    slice_nymex = df_slice[nymex_col].diff() * 100
    slice_rack = df_slice[rack_col].diff() * 100

    # Filter out roll and nan days
    valid_rows = []
    for i in range(len(df_slice)):
        dt = df_slice['date'].iloc[i]
        ch = slice_nymex.iloc[i]
        act = slice_rack.iloc[i]
        if pd.isna(ch) or pd.isna(act):
            continue
        if is_contract_roll_day(dt, prefix):
            continue
        valid_rows.append((ch, act))

    if valid_rows:
        valid_nymex = pd.Series([r[0] for r in valid_rows])
        valid_rack = pd.Series([r[1] for r in valid_rows])
    else:
        valid_nymex = pd.Series(dtype='float64')
        valid_rack = pd.Series(dtype='float64')

    # 1. NYMEX daily volatility (cents)
    nymex_std = float(valid_nymex.std()) if len(valid_nymex) > 1 else 1.0

    # 2. Performance metrics over the window using final smoothed thresholds
    alerts = 0
    correct = 0
    savings_list = []

    for ch, act in zip(valid_nymex, valid_rack):
        if ch >= smoothed_hike:
            alerts += 1
            savings_list.append(act)
            if act > 0:
                correct += 1
        elif ch <= smoothed_drop:
            alerts += 1
            savings_list.append(-act)
            if act < 0:
                correct += 1

    win_rate = float(correct / alerts) if alerts > 0 else 0.70 # baseline fallback
    avg_savings = float(np.mean(savings_list)) if alerts > 0 else 0.0

    # 3. 95% CVaR of rack loss on WAIT (DROP) signal days only.
    #
    # Rationale: "Waiting risk" is the cost incurred when the system signals DROP but the
    # rack rises anyway. We therefore filter valid_rows to days where the NYMEX move
    # crossed the DROP threshold (ch <= smoothed_drop), then compute the CVaR on the
    # resulting rack moves. A positive rack move on a WAIT day is a realized loss from
    # waiting; the 95th-percentile tail average is the true operational tail risk.
    #
    # Including all rack moves (old approach) contaminates the distribution with BUY-day
    # rack moves, understating the true tail risk on waiting decisions.
    #
    # Minimum 40 WAIT-day observations required so at least 2 data points (5% × 40)
    # fall above the 95th-percentile threshold for a stable CVaR estimate.
    wait_rack_moves = np.array([act for (ch, act) in valid_rows if ch <= smoothed_drop])
    if len(wait_rack_moves) >= 40:
        cvar_threshold_val = np.percentile(wait_rack_moves, 95)
        cvar_val = float(np.mean(wait_rack_moves[wait_rack_moves >= cvar_threshold_val]))
    else:
        cvar_val = 3.0  # default fallback (insufficient WAIT-signal history)

    # 4. Conviction-conditional metrics computed over the full history
    conviction_bins = {
        "high": {"alerts": 0, "correct": 0, "savings": []},
        "mod": {"alerts": 0, "correct": 0, "savings": []},
        "low": {"alerts": 0, "correct": 0, "savings": []}
    }
    
    # Starting points for sequential out-of-sample smoothing (Issue 4.1)
    curr_smoothed_h = None
    curr_smoothed_d = None
    
    for i in range(opt_W, len(df)):
        df_train = df.iloc[i - opt_W:i]
        df_today = df.iloc[i]
        
        # Exclude contract roll days from OOS conviction scoring (Issue 8.1)
        dt = df_today['date']
        if is_contract_roll_day(dt, prefix):
            continue
            
        train_nymex, train_rack = get_clean_deltas(df_train, nymex_col, rack_col)
        if len(train_nymex) < 20:
            continue
        h_t, d_t = train_thresholds(train_nymex, train_rack, opt_Hp, opt_Dp, clamp_bounds)
        
        # Sequentially smooth the thresholds using the same Blend Alpha
        if curr_smoothed_h is None:
            curr_smoothed_h = h_t
            curr_smoothed_d = d_t
        else:
            curr_smoothed_h = round(alpha * h_t + (1 - alpha) * curr_smoothed_h, 2)
            curr_smoothed_d = round(alpha * d_t + (1 - alpha) * curr_smoothed_d, 2)
        
        nymex_std_t = float(train_nymex.std()) if len(train_nymex) > 1 else 1.0
        
        ch = df_today['delta_nymex']
        act = df_today['delta_rack']
        
        if pd.isna(ch) or pd.isna(act):
            continue
            
        z = ch / nymex_std_t if nymex_std_t > 0 else 0.0
        abs_z = abs(z)
        
        if abs_z >= 1.5:
            bin_name = "high"
        elif abs_z >= 1.0:
            bin_name = "mod"
        else:
            bin_name = "low"
            
        triggered = False
        saved = 0.0
        is_correct = False
        
        if ch >= curr_smoothed_h:
            triggered = True
            saved = act
            is_correct = (act > 0)
        elif ch <= curr_smoothed_d:
            triggered = True
            saved = -act
            is_correct = (act < 0)
            
        if triggered:
            conviction_bins[bin_name]["alerts"] += 1
            conviction_bins[bin_name]["savings"].append(saved)
            if is_correct:
                conviction_bins[bin_name]["correct"] += 1
                
    for bin_name in ["high", "mod", "low"]:
        b = conviction_bins[bin_name]
        alerts_count = b["alerts"]
        if alerts_count >= 20:
            bin_win_rate = float(b["correct"] / alerts_count)
            bin_savings = float(np.mean(b["savings"]))
        else:
            bin_win_rate = -1.0
            bin_savings = 0.0
            
        cfg[f"{prefix}_{bin_name}_z_win_rate"] = round(bin_win_rate, 4)
        cfg[f"{prefix}_{bin_name}_z_savings"] = round(bin_savings, 4)
        cfg[f"{prefix}_{bin_name}_z_count"] = alerts_count

    # Save metrics to config
    cfg[f"{prefix}_nymex_daily_std"] = round(nymex_std, 4)
    cfg[f"{prefix}_historical_win_rate"] = round(win_rate, 4)
    cfg[f"{prefix}_wait_upside_cvar_95"] = round(cvar_val, 4)
    cfg[f"{prefix}_average_savings"] = round(avg_savings, 4)
    cfg[f"{prefix}_window_days"] = opt_W
    cfg[f"{prefix}_opt_Hp"] = opt_Hp
    cfg[f"{prefix}_opt_Dp"] = opt_Dp

    msg = f"Hike={smoothed_hike}c, Drop={smoothed_drop}c, Vol={nymex_std:.2f}c, CVaR={cvar_val:.2f}c"
    return cfg, msg, opt_W

def main():
    print("Starting walk-forward backtest engine...")
    validate_data.validate_all(DATA_DIR)
    cfg = load_config()

    if not os.path.exists(CSV_PATH):
        print("No CSV data found. Exiting.")
        sys.exit(0)

    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    # Filter out weekend rows first to avoid Monday diff data loss (Issue 1.2)
    df = df[df['date'].dt.dayofweek < 5].reset_index(drop=True)
    df = df.sort_values('date').reset_index(drop=True)

    min_rows = cfg.get("MIN_ROWS_FOR_TUNING", 30)
    if len(df) < min_rows:
        print(f"Insufficient data. Have {len(df)} rows, need {min_rows}. Exiting.")
        sys.exit(0)

    # Walk-forward optimization separately for RB and HO
    cfg, msg_rb, rb_win = run_optimization(df, 'nymex_rb', 'rack_u', 'RB', cfg)
    cfg, msg_ho, ho_win = run_optimization(df, 'nymex_ho', 'rack_d', 'HO', cfg)

    # For logging compatibility, store the RB window as ROLLING_WINDOW_DAYS
    cfg["ROLLING_WINDOW_DAYS"] = rb_win
    cfg["LAG_DAYS"] = 0 # lag is physically zero

    save_metrics_cache(cfg)

    local_now = pd.Timestamp.now(tz='America/Chicago')
    commit_msg = f"Auto-tune [{local_now.strftime('%Y-%m-%d')}]: Walk-Forward. RB({msg_rb}) HO({msg_ho})"
    print(commit_msg)
    git_commit_push(commit_msg)

if __name__ == "__main__":
    main()
