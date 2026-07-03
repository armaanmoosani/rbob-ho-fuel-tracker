import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest
import verify_statistics

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")

def wilson_confidence_interval(correct, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_variance = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = (centre_adjusted_probability - adjusted_variance) / denominator
    upper = (centre_adjusted_probability + adjusted_variance) / denominator
    return max(0.0, lower), min(1.0, upper)

def run_audit():
    print("==========================================================")
    print("         GRAVES PRICING ENGINE - STATISTICAL AUDIT")
    print("==========================================================")

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV database. Total rows found: {len(df)}")
    
    # Assert database length is 766 or correct historical size
    assert len(df) >= 700, f"Expected >700 rows, found {len(df)}"

    df_clean = df.dropna(subset=['nymex_rb', 'rack_u']).copy().reset_index(drop=True)
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['delta_nymex'] = df_clean['nymex_rb'].diff() * 100
    df_clean['delta_rack'] = df_clean['rack_u'].diff() * 100
    df_clean = df_clean.dropna(subset=['delta_nymex', 'delta_rack']).copy().reset_index(drop=True)

    nymex_vals = df_clean['delta_nymex'].values
    rack_vals = df_clean['delta_rack'].values

    # --- 1. Durbin-Watson Autocorrelation Check ---
    print("\n[CHECK 1] OLS Residual Autocorrelation (Durbin-Watson)")
    slope, intercept, r_value, p_value, std_err = stats.linregress(nymex_vals, rack_vals)
    predicted_rack = intercept + slope * nymex_vals
    residuals = rack_vals - predicted_rack
    
    # Calculate Durbin-Watson statistic
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"  - OLS Slope:      {slope:.4f}")
    print(f"  - OLS R-squared:  {r_value**2:.4f}")
    print(f"  - OLS P-value:    {p_value:.4e}")
    print(f"  - Durbin-Watson:  {dw_stat:.4f}")
    if 1.5 <= dw_stat <= 2.5:
        print("  => VERDICT: PASSED. Residuals show no significant autocorrelation (1.5-2.5 is acceptable).")
    else:
        print("  => WARNING: Residual autocorrelation detected! P-value may be artificially deflated.")

    # --- 2. Holdout Sample Size Audit & Wilson Confidence Interval ---
    print("\n[CHECK 2] Holdout Sample Size & Confidence Interval Audit")
    holdout_days = 60
    df_train = df_clean.iloc[:-holdout_days]
    df_test = df_clean.iloc[-holdout_days:].reset_index(drop=True)

    hike_t, drop_t, _, _ = verify_statistics.tune_thresholds(df_train, 'nymex_rb', 'rack_u')
    savings, precision, total_alerts, per_alert_savings = verify_statistics.run_simulation(
        df_test, hike_t, drop_t, 'nymex_rb', 'rack_u'
    )
    
    correct_alerts = int(round(precision * total_alerts))
    lower_w, upper_w = wilson_confidence_interval(correct_alerts, total_alerts)
    
    print(f"  - Holdout Period:    Last {holdout_days} cleaned days")
    print(f"  - Tuned Thresholds:  Hike={hike_t:.2f}c, Drop={drop_t:.2f}c")
    print(f"  - Total Active Alerts: {total_alerts}")
    print(f"  - Correct Alerts:      {correct_alerts}")
    print(f"  - Precision Estimate:  {precision:.2%}")
    print(f"  - 95% Wilson Interval: [{lower_w:.2%}, {upper_w:.2%}]")
    if total_alerts < 50:
        print("  => NOTE: Holdout count is small. Rely on the wide Wilson interval for true performance bounds.")
    else:
        print("  => VERDICT: Sample size is large enough to be statistically meaningful.")

    # --- 3. Sharpe-Equivalent Ratio of Savings ---
    print("\n[CHECK 3] Sharpe-Equivalent Ratio of Active Alert Savings")
    if len(per_alert_savings) > 1:
        avg_s = np.mean(per_alert_savings)
        std_s = np.std(per_alert_savings)
        sharpe = avg_s / std_s if std_s > 0 else 0.0
        print(f"  - Mean Savings per Alert: {avg_s:.2f} c/gal")
        print(f"  - Std Dev of Savings:     {std_s:.2f} c/gal")
        print(f"  - Sharpe-Equivalent:      {sharpe:.4f}")
        if sharpe >= 0.5:
            print("  => VERDICT: High-quality edge. Consistency is strong (Sharpe >= 0.5).")
        else:
            print("  => WARNING: Volatile edge. Average savings are heavily influenced by outliers (Sharpe < 0.5).")
    else:
        print("  - N/A: No alerts triggered to compute Sharpe ratio.")

    # --- 4. Heteroskedasticity (Breusch-Pagan equivalent check) ---
    print("\n[CHECK 4] Heteroskedasticity Check (Residual variance vs. futures volatility)")
    # Split the dataset into high-volatility and low-volatility halves
    rolling_vol = df_clean['delta_nymex'].rolling(window=20).std().fillna(method='bfill')
    med_vol = np.median(rolling_vol)
    
    high_vol_res = residuals[rolling_vol >= med_vol]
    low_vol_res = residuals[rolling_vol < med_vol]
    
    f_stat = np.var(high_vol_res) / np.var(low_vol_res)
    # F-test p-value
    p_bp = stats.f.sf(f_stat, len(high_vol_res)-1, len(low_vol_res)-1)
    print(f"  - Residual variance in high-volatility regime: {np.var(high_vol_res):.4f}")
    print(f"  - Residual variance in low-volatility regime:  {np.var(low_vol_res):.4f}")
    print(f"  - F-statistic ratio:                           {f_stat:.4f}")
    print(f"  - F-test P-value (heteroskedasticity test):    {p_bp:.4e}")
    if p_bp < 0.05:
        print("  => WARNING: Heteroskedasticity is present. Residual variance differs across volatility regimes.")
    else:
        print("  => VERDICT: Homoskedasticity holds. Residual variance is stable across regimes.")

    # --- 5. True 90-Day Out-of-Sample Holdout Year test ---
    print("\n[CHECK 5] True 90-Day Out-of-Sample Test (Never seen in parameter tuning)")
    # Remove the last 90 rows entirely
    df_calibration = df_clean.iloc[:-90].copy()
    df_oos_test = df_clean.iloc[-90:].copy().reset_index(drop=True)

    # Run optimization grid search on the calibration set
    print("Running parameters grid optimization on calibration set...")
    cfg = {
        "MIN_ROWS_FOR_TUNING": 30,
        "BLEND_ALPHA": 0.3,
        "RB_HIKE_THRESHOLD_CENTS": 1.0,
        "RB_DROP_THRESHOLD_CENTS": -1.0,
        "RB_nymex_daily_std": 1.0,
        "RB_historical_win_rate": 0.70,
        "RB_wait_upside_cvar_95": 3.0,
        "RB_average_savings": 0.0,
        "RB_window_days": 120
    }
    
    cfg_opt, msg_opt, opt_W = backtest.run_optimization(df_calibration, 'nymex_rb', 'rack_u', 'RB', cfg)
    opt_hike = cfg_opt['RB_HIKE_THRESHOLD_CENTS']
    opt_drop = cfg_opt['RB_DROP_THRESHOLD_CENTS']
    
    print(f"Optimal parameters trained: Win={opt_W}, Hike={opt_hike:.2f}c, Drop={opt_drop:.2f}c")
    
    # Run simulation on completely held-out 90 days
    oos_sav, oos_prec, oos_alerts, oos_savings_list = verify_statistics.run_simulation(
        df_oos_test, opt_hike, opt_drop, 'nymex_rb', 'rack_u'
    )
    
    print(f"True Out-of-Sample Performance on last 90 days:")
    print(f"  - Total alerts triggered: {oos_alerts}")
    print(f"  - OOS Precision:          {oos_prec:.2%}")
    print(f"  - OOS Cum. Savings:        {oos_sav:+.2f} c/gal")
    
    lower_oos, upper_oos = wilson_confidence_interval(int(round(oos_prec * oos_alerts)), oos_alerts)
    print(f"  - 95% Wilson Interval:    [{lower_oos:.2%}, {upper_oos:.2%}]")
    if oos_sav > 0:
        print("  => SUCCESS: Model generated positive out-of-sample savings on completely held-out data.")
    else:
        print("  => WARNING: Model lost money or had 0 savings on out-of-sample data.")

    print("\n==========================================")

if __name__ == "__main__":
    run_audit()
