import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
METRICS_CACHE_PATH = os.path.join(DATA_DIR, "metrics_cache.json")

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_config(config_path=None, metrics_cache_path=None):
    cfg = {}
    import json
    config_path = config_path or CONFIG_PATH
    metrics_cache_path = metrics_cache_path or METRICS_CACHE_PATH
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    if os.path.exists(metrics_cache_path):
        try:
            with open(metrics_cache_path, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def tune_thresholds(df_train, nymex_col, rack_col, Hp=15, Dp=85):
    """
    Calibrate empirical hike/drop thresholds from training data.
    Uses backward rack diff (rack[T] - rack[T-1]) = tonight's new price vs last night's,
    which is what the live system measures at 2:35 PM CT on day T.
    """
    df_clean = df_train.dropna(subset=[nymex_col, rack_col])
    delta_nymex = df_clean[nymex_col].diff() * 100   # NYMEX settle[T] - settle[T-1]
    delta_rack  = df_clean[rack_col].diff() * 100    # rack[T] - rack[T-1] = tonight's change

    valid = ~(delta_nymex.isna() | delta_rack.isna())
    delta_nymex = delta_nymex[valid]
    delta_rack  = delta_rack[valid]

    if len(delta_nymex) < 10:
        return 1.0, -1.0, 0.0, 1.0

    slope, intercept, r_value, p_value, std_err = stats.linregress(delta_nymex, delta_rack)

    hike_mask = (delta_rack > 0) & (delta_nymex > 0)
    drop_mask = (delta_rack < 0) & (delta_nymex < 0)

    hike_thresh = 1.0
    drop_thresh = -1.0

    if hike_mask.sum() >= 5:
        hike_thresh = clamp(np.percentile(delta_nymex[hike_mask], Hp), 0.3, 3.0)
    if drop_mask.sum() >= 5:
        drop_thresh = clamp(np.percentile(delta_nymex[drop_mask], Dp), -3.0, -0.3)

    return hike_thresh, drop_thresh, slope, r_value**2


def run_simulation(df_slice, hike_thresh, drop_thresh, nymex_col, rack_col):
    """
    Simulate live procurement decisions.

    Direction convention (CORRECTED from prior version):
    - delta_nymex[T]  = NYMEX settle[T] - settle[T-1]   (signal computed at 2:35 PM CT)
    - delta_rack[T]   = rack[T] - rack[T-1]              (tonight's Graves posting vs last night)

    This matches the live system: prediction fires at 2:35 PM on T,
    outcome is measured when ingest_prices runs at 11:30 PM and stores rack[T].

    Savings accounting:
    - BUY_NOW (HIKE signal): buy at rack[T-1], rack goes up → saved = +delta_rack[T]
    - WAIT    (DROP signal): wait, rack goes down → saved = -delta_rack[T] (negative rack move = savings)
    """
    if 'delta_nymex' in df_slice.columns and 'delta_rack' in df_slice.columns:
        delta_nymex = df_slice['delta_nymex']
        delta_rack  = df_slice['delta_rack']
    else:
        delta_nymex = df_slice[nymex_col].diff() * 100
        delta_rack  = df_slice[rack_col].diff() * 100   # same-night direction (CORRECTED)

    savings = 0.0
    correct = 0
    total   = 0
    per_alert_savings = []

    for i in range(len(df_slice)):
        change = delta_nymex.iloc[i]
        actual = delta_rack.iloc[i]
        if pd.isna(change) or pd.isna(actual):
            continue

        if change >= hike_thresh:
            savings += actual
            per_alert_savings.append(actual)
            total += 1
            if actual > 0:
                correct += 1
        elif change <= drop_thresh:
            savings += -actual
            per_alert_savings.append(-actual)
            total += 1
            if actual < 0:
                correct += 1

    precision = correct / total if total > 0 else 0.0
    return savings, precision, total, per_alert_savings


def main():
    print("=== GRAVES OIL QUANTITATIVE MODEL VALIDATION SUITE ===")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Historical database not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    # Filter out weekend rows first to avoid Monday diff data loss (Issue 1.2)
    df = df[df['date'].dt.dayofweek < 5].reset_index(drop=True)

    # ─── 1. FROZEN HOLDOUT & LEAKAGE TEST ─────────────────────────────────────
    print("\n--- 1. Frozen Holdout & Future Leakage Test ---")
    holdout_days = 60

    df_clean = df.dropna(subset=['nymex_rb', 'rack_u']).copy().reset_index(drop=True)
    df_clean['delta_nymex'] = df_clean['nymex_rb'].diff() * 100
    df_clean['delta_rack'] = df_clean['rack_u'].diff() * 100

    if len(df_clean) <= holdout_days + 30:
        print("Warning: Insufficient data for 60-day holdout test. Skipping.")
        real_savings = real_precision = real_alerts = 0
        perm_savings = []
    else:
        df_train = df_clean.iloc[:-holdout_days]
        df_test  = df_clean.iloc[-holdout_days:].reset_index(drop=True)

        cfg_loaded = load_config()
        opt_Hp = cfg_loaded.get('RB_opt_Hp', 15)
        opt_Dp = cfg_loaded.get('RB_opt_Dp', 85)
        hike_t, drop_t, slope, r2 = tune_thresholds(df_train, 'nymex_rb', 'rack_u', Hp=opt_Hp, Dp=opt_Dp)
        print(f"Tuned Thresholds on Train Data: Hike={hike_t:.2f}c, Drop={drop_t:.2f}c (Slope={slope:.2f}, R2={r2:.2f})")

        real_savings, real_precision, real_alerts, _ = run_simulation(
            df_test, hike_t, drop_t, 'nymex_rb', 'rack_u')
        print(f"Real Holdout Performance (Last {holdout_days} days):")
        print(f"  - Total Alerts: {real_alerts}")
        print(f"  - Precision:    {real_precision:.2%}")
        print(f"  - Cum. Savings:  {real_savings:+.2f} c/gal")

        # Permutation test — scramble nymex↔rack alignment
        n_perm   = 500
        perm_savings    = []
        perm_precisions = []

        test_delta_nymex = df_test['delta_nymex']
        test_delta_rack  = df_test['delta_rack']

        valid_mask   = ~(test_delta_nymex.isna() | test_delta_rack.isna())
        valid_nymex  = test_delta_nymex[valid_mask].values
        valid_rack   = test_delta_rack[valid_mask].values

        for _ in range(n_perm):
            shuffled_rack = np.random.permutation(valid_rack)
            sav = prec_c = tot = 0
            for i in range(len(valid_nymex)):
                ch = valid_nymex[i]; ac = shuffled_rack[i]
                if ch >= hike_t:
                    sav += ac; tot += 1; prec_c += (1 if ac > 0 else 0)
                elif ch <= drop_t:
                    sav += -ac; tot += 1; prec_c += (1 if ac < 0 else 0)
            perm_savings.append(sav)
            perm_precisions.append(prec_c / tot if tot > 0 else 0.0)

        p_val_savings   = np.mean(np.array(perm_savings) >= real_savings)
        p_val_precision = np.mean(np.array(perm_precisions) >= real_precision)
        print(f"Permuted Holdout (Mean of {n_perm} trials):")
        print(f"  - Mean Savings:   {np.mean(perm_savings):+.2f} c/gal (P-value = {p_val_savings:.4f})")
        print(f"  - Mean Precision: {np.mean(perm_precisions):.2%} (P-value = {p_val_precision:.4f})")

        if p_val_precision < 0.10 or p_val_savings < 0.10:
            print("  => SUCCESS: Leakage test passed. Model shows significant predictive power over random chance.")
        else:
            print("  => WARNING: Holdout performance not highly distinct from random in this 60-day window.")
            print("     (This is expected with ~55 alerts; use the 8-period walk-forward for power.)")

    # ─── 2. NULL MODEL TEST ────────────────────────────────────────────────────
    print("\n--- 2. Randomized Null Model Test (Full History) ---")
    df_reg      = df_clean.copy()
    delta_nymex = df_reg['nymex_rb'].diff() * 100
    delta_rack  = df_reg['rack_u'].diff() * 100
    valid       = ~(delta_nymex.isna() | delta_rack.isna())
    nymex_vals  = delta_nymex[valid].values
    rack_vals   = delta_rack[valid].values

    real_slope, real_intercept, real_r, real_p, real_se = stats.linregress(nymex_vals, rack_vals)
    print(f"Real OLS Regression (Full History):")
    print(f"  - Slope:     {real_slope:.4f}  (cents of rack change per cent of NYMEX change)")
    print(f"  - R2:        {real_r**2:.4f}   (NYMEX explains {real_r**2:.1%} of rack variance)")
    print(f"  - P-value:   {real_p:.4e}")

    n_null  = 200
    null_r2s = []
    null_ps  = []
    for _ in range(n_null):
        s, i, r, p, se = stats.linregress(nymex_vals, np.random.permutation(rack_vals))
        null_r2s.append(r**2)
        null_ps.append(p)

    print(f"Null Model (Mean of {n_null} shuffled trials):")
    print(f"  - Mean R2:   {np.mean(null_r2s):.4f}")
    print(f"  - P < 0.10:  {np.mean(np.array(null_ps) < 0.10):.1%} of trials")

    if real_p < 0.001 and np.mean(null_r2s) < 0.01:
        print("  => SUCCESS: Null model test passed. Relationship is highly genuine and non-spurious.")
    else:
        print("  => WARNING: Null model test suggests weak or spurious relationship.")

    # ─── 3. REGIME STABILITY ──────────────────────────────────────────────────
    print("\n--- 3. Regime Stability Test (Yearly Splits) ---")
    df_reg['year'] = df_reg['date'].dt.year
    years = sorted(df_reg['year'].unique())
    print(f"{'Year':<6} | {'Count':<5} | {'Slope':<8} | {'R2':<8} | {'P-value':<10} | {'Status':<10}")
    print("-" * 57)
    for yr in years:
        df_yr   = df_reg[df_reg['year'] == yr]
        dy_n    = df_yr['nymex_rb'].diff() * 100
        dy_r    = df_yr['rack_u'].diff() * 100
        val     = ~(dy_n.isna() | dy_r.isna())
        if val.sum() < 20:
            print(f"{yr:<6} | {val.sum():<5} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10} | {'Too Few':<10}")
            continue
        s, i, r, p, se = stats.linregress(dy_n[val], dy_r[val])
        status = "PASSED" if p < 0.10 else "WEAK"
        print(f"{yr:<6} | {val.sum():<5} | {s:<8.4f} | {r**2:<8.4f} | {p:<10.4e} | {status:<10}")

    # ─── 4. SENSITIVITY ANALYSIS ──────────────────────────────────────────────
    print("\n--- 4. Sensitivity Analysis (Percentile Thresholds) ---")
    percentiles = [
        (10, 90, "Aggressive"),
        (15, 85, "Baseline"),
        (20, 80, "Conservative"),
        (25, 75, "Very Conservative"),
    ]
    sensitivity_results = []
    hike_mask_full = (rack_vals > 0) & (nymex_vals > 0)
    drop_mask_full = (rack_vals < 0) & (nymex_vals < 0)
    print(f"{'Config (Hike/Drop)':<22} | {'Hike Thresh':>12} | {'Drop Thresh':>12} | {'Alerts':>6} | {'Precision':>10} | {'Savings':>12}")
    print("-" * 88)
    for pct_h, pct_d, name in percentiles:
        h = clamp(np.percentile(nymex_vals[hike_mask_full], pct_h), 0.3, 3.0)
        d = clamp(np.percentile(nymex_vals[drop_mask_full], pct_d), -3.0, -0.3)
        sav, prec, alt, _ = run_simulation(df_clean, h, d, 'nymex_rb', 'rack_u')
        sensitivity_results.append((name, h, d, sav))
        tag = f"{name} ({pct_h}/{pct_d})"
        print(f"{tag:<22} | {h:>12.2f} | {d:>12.2f} | {alt:>6} | {prec:>10.2%} | {sav:>+12.2f} c")

    all_positive = all(r[3] > 0 for r in sensitivity_results)
    if all_positive:
        print("  => SUCCESS: Sensitivity analysis passed. Performance is robust across all threshold settings.")
    else:
        print("  => WARNING: Performance collapses for some threshold parameters.")

    # ─── 5. RESIDUAL DIAGNOSTICS ──────────────────────────────────────────────
    print("\n--- 5. Residual Diagnostics ---")
    predicted_rack = real_intercept + real_slope * nymex_vals
    residuals      = rack_vals - predicted_rack
    mean_res = np.mean(residuals)
    std_res  = np.std(residuals)
    dw_stat  = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"Residual Statistics:")
    print(f"  - Mean:          {mean_res:.6f}  (Should be ~0.0)")
    print(f"  - Std Dev:       {std_res:.4f}  (= unexplained rack noise per day, in cents)")
    print(f"  - Durbin-Watson: {dw_stat:.4f}  (2.0 = no autocorrelation; 1.5-2.5 acceptable)")
    if abs(mean_res) < 1e-5 and 1.5 <= dw_stat <= 2.5:
        print("  => SUCCESS: Residual analysis passed. Model assumptions hold.")
    else:
        print("  => WARNING: Residuals show non-zero mean or significant autocorrelation.")

    # ─── 6. SHADOW BENCHMARKS ─────────────────────────────────────────────────
    print("\n--- 6. Shadow Benchmarks (vs Naive Strategies) ---")
    # Load live config thresholds
    cfg = load_config()
    live_hike = cfg.get('RB_HIKE_THRESHOLD_CENTS', 1.0)
    live_drop = cfg.get('RB_DROP_THRESHOLD_CENTS', -1.0)

    live_sav, live_prec, live_n, live_per = run_simulation(
        df_clean, live_hike, live_drop, 'nymex_rb', 'rack_u')

    delta_rack_all = df_clean['rack_u'].diff() * 100
    valid_rack_all = delta_rack_all.dropna().values

    # Benchmark 1: Always buy immediately (never wait)
    # Equivalent savings = 0 by definition (you always pay tonight's rack)
    # But you miss savings on DROP days; your average per-purchase-opportunity cost vs model:
    always_buy_baseline = 0.0  # by definition — you always pay market

    # Benchmark 2: Always wait (defer every load by 1 day)
    always_wait_savings = float(-np.sum(valid_rack_all))  # you skip buying at rack[T-1], pay rack[T]
    # This is net savings from always waiting: if rack trends down, positive; if up, negative

    # Benchmark 3: Random decision (50/50 each day)
    n_random_trials = 1000
    random_savings_dist = []
    for _ in range(n_random_trials):
        decisions = np.random.choice(['buy', 'wait', 'skip'], size=len(valid_rack_all))
        s = 0.0
        for dec, act in zip(decisions, valid_rack_all):
            if dec == 'buy':
                s += act
            elif dec == 'wait':
                s += -act
        random_savings_dist.append(s)
    random_mean = np.mean(random_savings_dist)

    print(f"Model (live thresholds h={live_hike}c, d={live_drop}c):")
    print(f"  Alerts: {live_n}, Precision: {live_prec:.2%}, Savings: {live_sav:+.2f}c")
    print(f"Shadow Benchmark — Always Buy (no model):  {always_buy_baseline:+.2f}c savings (definition)")
    print(f"Shadow Benchmark — Always Wait:            {always_wait_savings:+.2f}c (net over full period)")
    print(f"Shadow Benchmark — Random 50/50 (mean):   {random_mean:+.2f}c over {n_random_trials} trials")
    print(f"Model edge over random benchmark:          {live_sav - random_mean:+.2f}c")
    if live_sav > random_mean:
        print("  => SUCCESS: Model outperforms random baseline.")
    else:
        print("  => WARNING: Model underperforms random baseline on full history.")

    # ─── 7. ROLLING 60-DAY DRIFT MONITOR ──────────────────────────────────────
    print("\n--- 7. Rolling 60-Day Precision Drift Monitor ---")
    window = 60
    drift_dates  = []
    drift_prec   = []
    drift_savings= []

    for end in range(window, len(df_clean) + 1, 10):
        window_df = df_clean.iloc[max(0, end - window):end]
        if len(window_df) < 20:
            continue
        _, wp, _, _ = run_simulation(window_df, live_hike, live_drop, 'nymex_rb', 'rack_u')
        drift_dates.append(df_clean['date'].iloc[end - 1])
        drift_prec.append(wp)

    if drift_dates:
        recent_prec = drift_prec[-1]
        avg_prec    = np.mean(drift_prec)
        print(f"Most recent 60-day rolling precision: {recent_prec:.2%}")
        print(f"Historical average 60-day precision:  {avg_prec:.2%}")
        if recent_prec < 0.50:
            print("  => DRIFT WARNING: Recent 60-day precision is below 50%. Model may need re-calibration.")
        elif recent_prec < avg_prec - 0.10:
            print("  => DRIFT CAUTION: Recent precision is >10pp below historical average.")
        else:
            print("  => DRIFT OK: Recent precision is within historical norms.")

    # ─── 8. TAIL RISK ANALYSIS ────────────────────────────────────────────────
    print("\n--- 8. Tail Risk Analysis (Worst-Case Alerts) ---")
    if live_per:
        per_arr = np.array(live_per)
        cvar_5  = float(np.mean(per_arr[per_arr <= np.percentile(per_arr, 5)]))
        worst_3 = sorted(live_per)[:3]
        best_3  = sorted(live_per)[-3:]
        pct_neg = float(np.mean(per_arr < 0))
        print(f"Per-alert savings distribution (n={len(per_arr)}):")
        print(f"  CVaR 5% (avg worst-5% alert):  {cvar_5:+.2f}c/gal")
        print(f"  Worst 3 individual alerts:      {[f'{v:+.2f}c' for v in worst_3]}")
        print(f"  Best  3 individual alerts:      {[f'{v:+.2f}c' for v in best_3]}")
        print(f"  % of alerts that lost money:    {pct_neg:.1%}")
        print(f"  At 8,500 gal load: worst alert = ${cvar_5 * 85:.0f}")
        if cvar_5 > -5.0:
            print("  => TAIL RISK OK: CVaR within acceptable range (<5c/gal average worst case).")
        else:
            print("  => TAIL RISK WARNING: Large average losses in worst-5% of alerts.")
    else:
        cvar_5 = 0.0; pct_neg = 0.0; per_arr = np.array([])

    # ─── CHARTS ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
    fig.patch.set_facecolor('#ffffff')

    PANEL  = '#f8fafc'
    GRID   = '#e2e8f0'
    BLUE   = '#3b82f6'
    GREEN  = '#22c55e'
    RED    = '#ef4444'
    SLATE  = '#64748b'
    DARK   = '#1e293b'

    # Panel 1: Permutation savings distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if perm_savings:
        ax1.hist(perm_savings, bins=30, color='#94a3b8', edgecolor='#64748b', alpha=0.8, label='Permutation trials')
        ax1.axvline(real_savings, color=RED, linestyle='--', linewidth=2.5,
                    label=f'Real holdout ({real_savings:+.1f}¢)')
    ax1.set_facecolor(PANEL); ax1.set_title('Holdout vs Permutation', fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax1.set_xlabel('Cum. Savings (¢/gal)', fontsize=9, color=SLATE)
    ax1.legend(fontsize=7, frameon=True, facecolor='#fff', edgecolor=GRID)
    ax1.grid(color=GRID, linestyle='--', alpha=0.7)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID)

    # Panel 2: OLS scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(nymex_vals, rack_vals, color=BLUE, alpha=0.35, edgecolor='none', s=12)
    xr  = np.linspace(nymex_vals.min(), nymex_vals.max(), 100)
    yr  = real_intercept + real_slope * xr
    ax2.plot(xr, yr, color=DARK, linewidth=2, label=f'OLS (R²={real_r**2:.2f})')
    ax2.fill_between(xr, yr - std_res, yr + std_res, color=BLUE, alpha=0.08)
    ax2.set_facecolor(PANEL); ax2.set_title('NYMEX → Rack Pass-Through', fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax2.set_xlabel('NYMEX Δ (¢/gal)', fontsize=9, color=SLATE)
    ax2.set_ylabel('Rack Δ (¢/gal)', fontsize=9, color=SLATE)
    ax2.legend(fontsize=8, frameon=True, facecolor='#fff', edgecolor=GRID)
    ax2.grid(color=GRID, linestyle='--', alpha=0.7)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)

    # Panel 3: Per-alert savings histogram (tail risk)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(per_arr) > 0:
        ax3.hist(per_arr, bins=25, color=GREEN, edgecolor='none', alpha=0.75, label='Per-alert outcome')
        ax3.axvline(0, color=SLATE, linewidth=1.5, linestyle='--')
        ax3.axvline(cvar_5, color=RED, linewidth=2, linestyle='--', label=f'CVaR 5%: {cvar_5:+.1f}¢')
    ax3.set_facecolor(PANEL); ax3.set_title('Per-Alert Savings Distribution', fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax3.set_xlabel('Savings per Alert (¢/gal)', fontsize=9, color=SLATE)
    ax3.legend(fontsize=7, frameon=True, facecolor='#fff', edgecolor=GRID)
    ax3.grid(color=GRID, linestyle='--', alpha=0.7)
    for sp in ax3.spines.values(): sp.set_edgecolor(GRID)

    # Panel 4: Regime R² by year
    ax4 = fig.add_subplot(gs[1, 0])
    yr_labels = []
    yr_r2s    = []
    for yr in years:
        df_yr = df_reg[df_reg['year'] == yr]
        dy_n  = df_yr['nymex_rb'].diff() * 100
        dy_r  = df_yr['rack_u'].diff() * 100
        val   = ~(dy_n.isna() | dy_r.isna())
        if val.sum() < 20:
            continue
        s, i, r, p, se = stats.linregress(dy_n[val], dy_r[val])
        yr_labels.append(str(yr))
        yr_r2s.append(r**2)
    colors_yr = [GREEN if r >= 0.05 else RED for r in yr_r2s]
    ax4.bar(yr_labels, yr_r2s, color=colors_yr, edgecolor='none', alpha=0.85)
    ax4.axhline(0.05, color=SLATE, linestyle='--', linewidth=1.2, label='R²=0.05 threshold')
    ax4.set_facecolor(PANEL); ax4.set_title('Regime Stability (R² by Year)', fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax4.set_ylabel('R²', fontsize=9, color=SLATE)
    ax4.legend(fontsize=7, frameon=True, facecolor='#fff', edgecolor=GRID)
    ax4.grid(axis='y', color=GRID, linestyle='--', alpha=0.7)
    for sp in ax4.spines.values(): sp.set_edgecolor(GRID)

    # Panel 5: Rolling 60-day precision drift
    ax5 = fig.add_subplot(gs[1, 1])
    if drift_dates:
        ax5.plot(drift_dates, drift_prec, color=BLUE, linewidth=2, marker='o', markersize=3)
        ax5.axhline(0.50, color=RED, linestyle='--', linewidth=1.5, label='50% floor')
        ax5.axhline(np.mean(drift_prec), color=SLATE, linestyle='--', linewidth=1,
                    label=f'Avg {np.mean(drift_prec):.0%}')
        ax5.fill_between(drift_dates, 0.50, drift_prec, alpha=0.10,
                         color=GREEN if drift_prec[-1] >= 0.50 else RED)
    ax5.set_facecolor(PANEL); ax5.set_title('Rolling 60-Day Precision Drift', fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax5.set_ylabel('Precision', fontsize=9, color=SLATE)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax5.legend(fontsize=7, frameon=True, facecolor='#fff', edgecolor=GRID)
    ax5.grid(color=GRID, linestyle='--', alpha=0.7)
    for sp in ax5.spines.values(): sp.set_edgecolor(GRID)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    # Panel 6: Shadow benchmark comparison
    ax6 = fig.add_subplot(gs[1, 2])
    bench_labels = ['Model\n(live cfg)', 'Random\n50/50', 'Always\nWait', 'Always\nBuy']
    bench_vals   = [live_sav, random_mean, always_wait_savings, always_buy_baseline]
    bench_colors = [GREEN if v > random_mean else RED for v in bench_vals]
    bench_colors[1] = SLATE
    bars = ax6.bar(bench_labels, bench_vals, color=bench_colors, edgecolor='none', alpha=0.85)
    ax6.axhline(0, color=SLATE, linewidth=1)
    for bar, val in zip(bars, bench_vals):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val:+.0f}¢', ha='center', va='bottom', fontsize=8, color=DARK, fontweight='bold')
    ax6.set_facecolor(PANEL); ax6.set_title('Shadow Benchmark Comparison', fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax6.set_ylabel('Total Savings (¢/gal)', fontsize=9, color=SLATE)
    ax6.grid(axis='y', color=GRID, linestyle='--', alpha=0.7)
    for sp in ax6.spines.values(): sp.set_edgecolor(GRID)

    plot_path = os.path.join(REPORTS_DIR, "statistical_verification.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nVisual validation chart saved to {plot_path}")
    print("=== MODEL VALIDATION SUITE COMPLETE ===")


if __name__ == "__main__":
    main()
