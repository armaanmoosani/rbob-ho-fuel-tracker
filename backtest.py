import os
import sys
import json
import hashlib
import subprocess
import pandas as pd
import numpy as np
import validate_data
from futures_util import is_contract_roll_day, is_nymex_business_day
from calibration_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    append_calibration_artifact,
    artifact_id,
    load_calibration_artifacts,
)
import pytz
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
METRICS_CACHE_PATH = os.path.join(DATA_DIR, "metrics_cache.json")
CALIBRATION_RUNS_PATH = os.path.join(DATA_DIR, "calibration_runs.jsonl")
CALIBRATION_PURGE_ROWS = 1
CALIBRATION_GRID_VERSION = "v1-purged-three-fold"

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
        paths = ["data/metrics_cache.json", "data/integrity_hashes.csv"]
        if os.path.exists(CALIBRATION_RUNS_PATH):
            paths.append("data/calibration_runs.jsonl")
        subprocess.run(["git", "add", *paths], check=True)
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

def build_purged_walk_forward_folds(row_count, window_days, test_size=90,
                                    fold_count=3, purge_rows=CALIBRATION_PURGE_ROWS):
    """Return chronological, non-overlapping train/purge/test boundaries.

    Indices use normal Python half-open intervals.  The purge separates every
    training endpoint from the following test block, including in the most
    recent fold, so a rack outcome cannot influence the threshold that scores
    it.
    """
    if purge_rows < 1:
        raise ValueError("purge_rows must be at least one session")
    folds = []
    for fold_number in range(fold_count):
        test_start = row_count - (fold_number + 1) * test_size
        test_end = row_count - fold_number * test_size if fold_number else row_count
        train_end = test_start - purge_rows
        train_start = train_end - window_days
        if train_start < 0:
            return []
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "purge_start": train_end,
            "purge_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
        })
    return list(reversed(folds))


def simulate_walk_forward(df, nymex_col, rack_col, W, Hp, Dp, prefix,
                          clamp_bounds=None, purge_rows=CALIBRATION_PURGE_ROWS):
    """
    Simulates walk-forward out-of-sample testing on 3 folds.
    Returns the median savings across all folds.
    """
    # Callers outside run_optimization may pass raw prices.  Compute once over
    # the complete series so the first test row retains its prior-session move.
    df = df.copy()
    if "delta_nymex" not in df.columns:
        df["delta_nymex"] = df[nymex_col].diff() * 100
    if "delta_rack" not in df.columns:
        df["delta_rack"] = df[rack_col].diff() * 100

    # 3 folds, 90-day test window each
    folds = 3
    test_size = 90
    fold_boundaries = build_purged_walk_forward_folds(
        len(df), W, test_size, folds, purge_rows
    )
    if not fold_boundaries:
        return -9999.0

    fold_savings = []

    for boundary in fold_boundaries:
        df_train = df.iloc[boundary["train_start"]:boundary["train_end"]]
        df_test = df.iloc[boundary["test_start"]:boundary["test_end"]]

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

def run_optimization(df, nymex_col, rack_col, prefix, cfg, windows=None,
                     hike_percentiles=None, drop_percentiles=None):
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

    windows = windows or [90, 120, 180, 240]
    hike_percentiles = hike_percentiles or [10, 15, 20, 25]
    drop_percentiles = drop_percentiles or [75, 80, 85, 90]

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


def _next_nymex_business_session(day):
    session = pd.Timestamp(day).date() + pd.Timedelta(days=1)
    while not is_nymex_business_day(session):
        session += pd.Timedelta(days=1)
    return session.isoformat()


def _history_hash(df):
    """Stable hash of exactly the rows eligible to train one artifact."""
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    payload = normalized.to_csv(index=False, lineterminator="\n", float_format="%.10f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _calibration_payload(cfg):
    keys = [
        "ROLLING_WINDOW_DAYS", "LAG_DAYS", "BLEND_ALPHA",
        "CLAMP_HIKE_MIN", "CLAMP_HIKE_MAX", "CLAMP_DROP_MIN", "CLAMP_DROP_MAX",
    ]
    keys.extend(key for key in cfg if key.startswith("RB_") or key.startswith("HO_"))
    return {key: cfg[key] for key in sorted(set(keys)) if key in cfg}


def build_shadow_calibration_artifact(df, cfg, effective_session=None,
                                      prior_artifact=None, windows=None,
                                      hike_percentiles=None, drop_percentiles=None):
    """Build, but do not persist, the calibration eligible for the next session.

    The artifact includes every rack outcome known when it is produced.  The
    one-session purge is applied inside its historical validation folds, where
    it prevents an evaluated row from influencing the preceding fit.  Excluding
    a fully received nightly rack row from tomorrow's calibration would add an
    unnecessary one-session delay without preventing leakage.
    """
    clean = df.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean[clean["date"].dt.dayofweek < 5].sort_values("date").reset_index(drop=True)
    if len(clean) <= CALIBRATION_PURGE_ROWS:
        raise ValueError("Not enough history to create a purged calibration artifact.")

    requested_session = pd.Timestamp(effective_session).date() if effective_session else None
    matching_sessions = clean.index[clean["date"].dt.date == requested_session].tolist() \
        if requested_session else []
    if matching_sessions:
        # Historical replay artifact: only outcomes strictly before the decision
        # session were knowable at the decision time.
        training_end_index = matching_sessions[0] - 1
    else:
        # Nightly production artifact for the next business session: today's
        # completed rack outcome is already known and is eligible.
        training_end_index = len(clean) - 1
    if training_end_index < 0:
        raise ValueError("No completed rack outcome exists before this calibration session.")
    training_df = clean.iloc[:training_end_index + 1].copy()
    effective_session = effective_session or _next_nymex_business_session(clean["date"].iloc[-1])

    artifact_cfg = dict(cfg)
    if prior_artifact:
        artifact_cfg.update(prior_artifact["calibration"])

    artifact_cfg, _, rb_window = run_optimization(
        training_df, "nymex_rb", "rack_u", "RB", artifact_cfg,
        windows=windows, hike_percentiles=hike_percentiles,
        drop_percentiles=drop_percentiles,
    )
    artifact_cfg, _, ho_window = run_optimization(
        training_df, "nymex_ho", "rack_d", "HO", artifact_cfg,
        windows=windows, hike_percentiles=hike_percentiles,
        drop_percentiles=drop_percentiles,
    )
    artifact_cfg["ROLLING_WINDOW_DAYS"] = rb_window
    artifact_cfg["LAG_DAYS"] = 0

    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "effective_session": str(effective_session)[:10],
        "training_start": training_df["date"].iloc[0].date().isoformat(),
        "training_end": training_df["date"].iloc[-1].date().isoformat(),
        "purge_rows": CALIBRATION_PURGE_ROWS,
        "source_history_hash": _history_hash(training_df),
        "source_row_count": len(training_df),
        "candidate_grid_version": CALIBRATION_GRID_VERSION,
        "objective": "median_out_of_sample_savings_cents; three 90-session purged folds",
        "prior_artifact_id": prior_artifact["artifact_id"] if prior_artifact else "bootstrap_config",
        "calibration": _calibration_payload(artifact_cfg),
        "generated_at": datetime.now(pytz.timezone("America/Chicago")).isoformat(),
    }
    artifact["artifact_id"] = artifact_id(artifact)
    return artifact


def write_shadow_calibration_artifact(df, cfg):
    """Persist one immutable next-session artifact without changing live inputs."""
    artifacts = load_calibration_artifacts(CALIBRATION_RUNS_PATH)
    dates = pd.to_datetime(df["date"])
    latest_session = dates[dates.dt.dayofweek < 5].max()
    effective_session = _next_nymex_business_session(latest_session)
    existing_index = next(
        (index for index, item in enumerate(artifacts)
         if item["effective_session"] == effective_session),
        None,
    )
    # A rerun must reproduce the original state transition.  In particular, it
    # cannot smooth against the artifact it is attempting to verify.
    prior = artifacts[existing_index - 1] if existing_index and existing_index > 0 \
        else (artifacts[-1] if existing_index is None and artifacts else None)
    artifact = build_shadow_calibration_artifact(
        df, cfg, effective_session=effective_session, prior_artifact=prior
    )
    written, created = append_calibration_artifact(CALIBRATION_RUNS_PATH, artifact)
    action = "Created" if created else "Verified existing"
    print(f"{action} shadow calibration artifact {written['artifact_id'][:12]} "
          f"for {written['effective_session']} (training through {written['training_end']}).")
    return written, created

def main():
    print("Starting walk-forward backtest engine...")
    validate_data.validate_all(DATA_DIR)
    cfg = load_config()
    calibration_seed_cfg = dict(cfg)

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

    # Shadow-only until a full calibration window has accumulated.  Live tracker
    # thresholds remain on the established cache path during this validation run.
    write_shadow_calibration_artifact(df, calibration_seed_cfg)
    validate_data.validate_calibration_artifacts(CALIBRATION_RUNS_PATH)
    validate_data.validate_and_update_hashes(DATA_DIR)

    local_now = pd.Timestamp.now(tz='America/Chicago')
    commit_msg = f"Auto-tune [{local_now.strftime('%Y-%m-%d')}]: Walk-Forward. RB({msg_rb}) HO({msg_ho})"
    print(commit_msg)
    git_commit_push(commit_msg)

if __name__ == "__main__":
    main()
