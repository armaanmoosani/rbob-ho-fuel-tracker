import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import validate_data
from scipy.stats import norm
import pytz
import json

def mask_recipient(address):
    if not address:
        return "None"
    if isinstance(address, (list, tuple)):
        return [mask_recipient(x) for x in address]
    address = str(address)
    if ',' in address:
        return [mask_recipient(x.strip()) for x in address.split(',') if x.strip()]
    if '@' in address:
        parts = address.split('@')
        name = parts[0]
        domain = parts[1]
        if len(name) <= 3:
            masked_name = "***"
        else:
            masked_name = name[:2] + "***" + name[-1:]
        return f"{masked_name}@{domain}"
    else:
        if len(address) <= 4:
            return "***"
        return address[:2] + "***" + address[-2:]

def mask_sensitive_text(text):
    if not text:
        return ""
    text = str(text)
    sensitive_vals = []
    for env_var in ['GMAIL_USER', 'GRAVES_EMAIL', 'TO_EMAIL']:
        val = os.environ.get(env_var, '')
        if val:
            for item in val.split(','):
                item_stripped = item.strip()
                if item_stripped and len(item_stripped) > 2:
                    sensitive_vals.append(item_stripped)
    sensitive_vals = sorted(list(set(sensitive_vals)), key=len, reverse=True)
    for val in sensitive_vals:
        text = text.replace(val, mask_recipient(val))
    return text


def mann_kendall_test(x):
    n = len(x)
    if n < 10:
        return False, 1.0, 0.0, 0.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    unique_x, counts = np.unique(x, return_counts=True)
    tie_sum = 0
    for t in counts:
        if t > 1:
            tie_sum += t * (t - 1) * (2 * t + 5)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    sens_slope = np.median(slopes) if slopes else 0.0
    tau = s / (0.5 * n * (n - 1))
    return (p_value < 0.05), p_value, sens_slope, tau



DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
TZ = pytz.timezone('America/Chicago')

os.makedirs(REPORTS_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

def load_config():
    defaults = {
        "DISPATCH_SAME_DAY_RATE": 0.50
    }
    cfg = defaults.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                user_cfg = json.load(f)
                cfg.update(user_cfg)
        except Exception:
            pass
    return cfg


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Historical database missing at: {CSV_PATH}")
        return
    validate_data.validate_all(DATA_DIR)
    if not os.path.exists(LOG_PATH):
        print(f"No prediction log found at: {LOG_PATH}. Waiting for first prediction run.")
        return

    log_df = pd.read_csv(LOG_PATH)
    hist_df = pd.read_csv(CSV_PATH)

    if len(log_df) == 0:
        print("No predictions logged yet.")
        return

    # Backfill PENDING outcomes
    updates_made = False
    hist_date_to_idx = {date_str: idx for idx, date_str in enumerate(hist_df['date'])}
    for idx, row in log_df.iterrows():
        if str(row['actual_next_day_move_cents']) == 'PENDING':
            pred_date = row['timestamp'].split('T')[0]
            # Find pred_date in hist_df (= day T, when prediction was made)
            hist_idx = hist_date_to_idx.get(pred_date)
            if hist_idx is not None and hist_idx - 1 >= 0:
                prev_idx = hist_idx - 1
                curr_row = hist_df.iloc[hist_idx]   # rack[T] = tonight's new price
                prev_row = hist_df.iloc[prev_idx]       # rack[T-1] = last night's price
                
                rack_col = 'rack_u' if row['commodity'] == 'RB' else 'rack_d'
                
                if pd.notna(curr_row[rack_col]) and pd.notna(prev_row[rack_col]):
                    # rack[T] - rack[T-1]: positive = rack went up (hike was correct)
                    move = (curr_row[rack_col] - prev_row[rack_col]) * 100
                    log_df.at[idx, 'actual_next_day_move_cents'] = round(move, 2)
                    updates_made = True

    if updates_made:
        log_df.to_csv(LOG_PATH, index=False)
        print("Backfilled PENDING outcomes.")

    # Calculate Metrics
    # Filter to only rows that have been resolved
    df = log_df[log_df['actual_next_day_move_cents'] != 'PENDING'].copy()
    df['actual_move'] = pd.to_numeric(df['actual_next_day_move_cents'], errors='coerce')
    df = df.dropna(subset=['actual_move']).copy()
    if len(df) == 0:
        print("No resolved predictions yet.")
        return
    
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True).dt.tz_convert('America/Chicago')
    df = df.sort_values('timestamp_dt').copy()

    # Vectorized calculation of savings and correctness
    pred = df['predicted_direction']
    actual = df['actual_move']

    df['savings_cents'] = np.select(
        [pred == 'HIKE', pred == 'DROP'],
        [actual, -actual],
        default=0.0
    )

    df['is_correct'] = np.select(
        [pred == 'HIKE', pred == 'DROP'],
        [actual > 0, actual < 0],
        default=False
    )

    df['cumulative_savings'] = df['savings_cents'].cumsum()

    # Pre-calculate active alert metrics using direct sum filters
    correct_hikes = int(((pred == 'HIKE') & (actual > 0)).sum())
    correct_drops = int(((pred == 'DROP') & (actual < 0)).sum())
    false_flat = int((((pred == 'HIKE') | (pred == 'DROP')) & (actual == 0)).sum())
    false_wrong_dir = int(((pred == 'HIKE') & (actual < 0)).sum() + ((pred == 'DROP') & (actual > 0)).sum())
    missed_moves = int(((pred == 'FLAT') & (actual.abs() > 0)).sum())
    
    # Correct total active alerts count (only HIKE and DROP signals)
    df_alerts = df[df['predicted_direction'].isin(['HIKE', 'DROP'])].copy()
    total_active_alerts = len(df_alerts)
    
    cfg = load_config()
    dispatch_rate = cfg.get("DISPATCH_SAME_DAY_RATE", 0.50)
    
    # Lifetime metrics
    lifetime_precision = (df_alerts['is_correct'].sum() / total_active_alerts * 100) if total_active_alerts > 0 else 0.0
    lifetime_savings_cents = df_alerts['savings_cents'].sum()
    avg_savings_per_active_alert_cents = (lifetime_savings_cents / total_active_alerts) if total_active_alerts > 0 else 0.0
    
    TRUCK_GALLONS = 8500
    lifetime_savings_dollars = (lifetime_savings_cents / 100.0) * TRUCK_GALLONS
    est_realized_lifetime_dollars = lifetime_savings_dollars * dispatch_rate
    avg_savings_per_truck_dollars = (avg_savings_per_active_alert_cents / 100.0) * TRUCK_GALLONS
    
    # Weekly metrics (Last 7 days activity)
    now_chicago = pd.Timestamp.now(tz='America/Chicago')
    cutoff_7d = now_chicago - pd.Timedelta(days=7)
    
    # Filter resolved predictions in the last 7 calendar days
    df_week = df[df['timestamp_dt'] >= cutoff_7d].copy()
    df_week_alerts = df_week[df_week['predicted_direction'].isin(['HIKE', 'DROP'])].copy()
    
    week_active_fired = len(df_week_alerts)
    week_correct = df_week_alerts['is_correct'].sum() if week_active_fired > 0 else 0
    week_precision = (week_correct / week_active_fired * 100) if week_active_fired > 0 else 0.0
    week_savings_cents = df_week_alerts['savings_cents'].sum() if week_active_fired > 0 else 0.0
    week_savings_dollars = (week_savings_cents / 100.0) * TRUCK_GALLONS
    est_realized_week_dollars = week_savings_dollars * dispatch_rate

    # Format Recent Resolved Alerts Log for last 7 days
    recent_alerts_html = ""
    if len(df_week) == 0:
        recent_alerts_html = "<tr><td colspan='6' style='text-align: center; padding: 16px; color: #64748b; font-size: 13px;'>No alerts resolved in the last 7 days.</td></tr>"
    else:
        # Sort newest first
        df_week_sorted = df_week.sort_values('timestamp_dt', ascending=False)
        for idx, row in df_week_sorted.iterrows():
            date_str = row['timestamp_dt'].strftime("%Y-%m-%d")
            comm_label = "Gas (RBOB)" if row['commodity'] == 'RB' else "Diesel (HO)"
            
            # Signal
            pred = row['predicted_direction']
            if pred == 'HIKE':
                sig_html = "<span style='color: #22c55e; font-weight: bold;'>BUY (Hike)</span>"
            elif pred == 'DROP':
                sig_html = "<span style='color: #f59e0b; font-weight: bold;'>WAIT (Drop)</span>"
            else:
                sig_html = "<span style='color: #64748b;'>FLAT (None)</span>"
                
            # Actual Move
            act_move = row['actual_move']
            act_move_str = f"{act_move:+.2f}¢/gal"
            
            # Result
            if pred == 'FLAT':
                res_str = "Flat (Suppressed)"
                res_color = "#64748b"
            else:
                if row['is_correct']:
                    res_str = "Correct"
                    res_color = "#16a34a"
                else:
                    res_str = "False Alarm"
                    res_color = "#ef4444"
            res_html = f"<span style='color: {res_color}; font-weight: 500;'>{res_str}</span>"
            
            # Savings
            sav_val = row['savings_cents']
            if pred == 'FLAT':
                sav_str = "—"
                sav_color = "#64748b"
            else:
                sav_str = f"{sav_val:+.2f}¢/gal"
                sav_color = "#16a34a" if sav_val >= 0 else "#ef4444"
            sav_html = f"<span style='color: {sav_color}; font-weight: bold;'>{sav_str}</span>"
            
            recent_alerts_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px 8px; color: #475569; font-size: 13px;">{date_str}</td>
                <td style="padding: 10px 8px; color: #475569; font-size: 13px;">{comm_label}</td>
                <td style="padding: 10px 8px; font-size: 13px;">{sig_html}</td>
                <td style="padding: 10px 8px; color: #475569; font-size: 13px;">{act_move_str}</td>
                <td style="padding: 10px 8px; font-size: 13px;">{res_html}</td>
                <td style="padding: 10px 8px; font-size: 13px; text-align: right;">{sav_html}</td>
            </tr>
            """
            
    # Check if we have 90 days of prediction history
    if len(df) > 0:
        has_90d_history = (df['timestamp_dt'].max() - df['timestamp_dt'].min()) >= pd.Timedelta(days=90)
    else:
        has_90d_history = False

    # Filter to active alerts (HIKE/DROP) for rolling precision calculation
    df_alerts = df_alerts.copy()
    df_alerts['rolling_precision'] = np.nan  # initialize so column always exists
    if has_90d_history and len(df_alerts) > 0:
        rolling_precisions = []
        for i, row in df_alerts.iterrows():
            t = row['timestamp_dt']
            window_start = t - pd.Timedelta(days=90)
            window = df_alerts[(df_alerts['timestamp_dt'] >= window_start) & (df_alerts['timestamp_dt'] <= t)]
            if len(window) > 0:
                prec = window['is_correct'].sum() / len(window)
            else:
                prec = np.nan
            rolling_precisions.append(prec * 100) # percentage
        df_alerts['rolling_precision'] = rolling_precisions

    # Weekly Alert Frequency Anomaly Monitoring
    df['week'] = df['timestamp_dt'].dt.to_period('W')
    alerts_per_week = df[df['predicted_direction'].isin(['HIKE', 'DROP'])].groupby('week').size()
    mean_alerts = float(alerts_per_week.mean()) if not alerts_per_week.empty else 0.0
    std_alerts = float(alerts_per_week.std()) if not alerts_per_week.empty and len(alerts_per_week) > 1 else 0.0

    current_week_start = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(days=7)
    raw_current_alerts = len(df[(df['timestamp_dt'] >= current_week_start) & (df['predicted_direction'].isin(['HIKE', 'DROP']))])

    # Count roll days in the last 7 calendar days
    from futures_util import is_contract_roll_day
    def get_roll_days_count(start_dt, end_dt):
        count = 0
        try:
            for dt in pd.date_range(start_dt.date(), end_dt.date()):
                if dt.weekday() < 5:
                    if is_contract_roll_day(dt, 'RB') or is_contract_roll_day(dt, 'HO'):
                        count += 1
        except Exception:
            pass
        return count

    roll_days_this_week = get_roll_days_count(current_week_start, pd.Timestamp.now(tz='America/Chicago'))
    if roll_days_this_week < 5:
        normalized_current_alerts = raw_current_alerts * (5 / (5 - roll_days_this_week))
    else:
        normalized_current_alerts = raw_current_alerts

    anomaly_threshold = mean_alerts + 2 * std_alerts if std_alerts > 0 else 999.0
    frequency_anomaly = (normalized_current_alerts > anomaly_threshold)

    freq_note = f"Normal frequency ({raw_current_alerts} alerts fired this week, normalized: {normalized_current_alerts:.1f} vs historical mean: {mean_alerts:.1f}/week)."
    if frequency_anomaly:
        freq_note = f"WARNING: Statistically unusual alert frequency detected! {raw_current_alerts} alerts fired this week (normalized: {normalized_current_alerts:.1f} vs historical limit: {anomaly_threshold:.1f}/week). This may indicate threshold miscalibration or extreme volatility."

    # Basis Stability Check using Mann-Kendall Trend Test
    basis_results = {}
    basis_warning = ""
    if os.path.exists(CSV_PATH):
        try:
            hist_df = pd.read_csv(CSV_PATH)
            rb_clean = hist_df.dropna(subset=['rack_u', 'nymex_rb']).copy()
            ho_clean = hist_df.dropna(subset=['rack_d', 'nymex_ho']).copy()
            
            rb_clean['basis'] = (rb_clean['rack_u'] - rb_clean['nymex_rb']) * 100
            ho_clean['basis'] = (ho_clean['rack_d'] - ho_clean['nymex_ho']) * 100
            
            for prefix, df_c in [('RB', rb_clean), ('HO', ho_clean)]:
                if len(df_c) >= 65:
                    last_90d_basis = df_c['basis'].tail(65).values
                    drifted, p_val, slope, tau = mann_kendall_test(last_90d_basis)
                    basis_results[prefix] = {
                        "drift": drifted,
                        "p_value": p_val,
                        "slope": slope,
                        "tau": tau,
                        "note": f"Rolling 90-day basis: Kendall tau = {tau:+.3f} (p = {p_val:.3f}, slope = {slope*65:+.2f}¢/gal over 90 days)."
                    }
                    if drifted:
                        basis_warning += f"WARNING: Significant basis drift detected in {prefix} (Kendall tau: {tau:+.3f}, p = {p_val:.3f}). Pricing thresholds may require manual review.<br>"
                else:
                    basis_results[prefix] = {
                        "drift": False,
                        "note": "Insufficient historical data for basis stability check."
                    }
        except Exception as e:
            print(f"Error checking basis stability: {e}")

    # Stable 180-Day Permutation Significance Testing
    # Filter to last 180 days Chicago time (to avoid time zone differences)
    cutoff_date = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(days=180)
    df_180 = df[df['timestamp_dt'] >= cutoff_date].copy()
    
    # If not enough data in last 180 days, fall back to full resolved dataset
    if len(df_180) < 5:
        df_sig = df
        sig_period_str = "Full History"
    else:
        df_sig = df_180
        sig_period_str = "Last 180 Days"
        
    p_value = 1.0
    p_value_note = "Model significance could not be computed (insufficient resolved predictions)."
    p_value_str = "N/A"
    
    if len(df_sig) >= 5:
        real_sig_savings = 0.0
        pred_dirs = []
        actual_moves = []
        
        for idx, row in df_sig.iterrows():
            pred = row['predicted_direction']
            act = row['actual_move']
            pred_dirs.append(pred)
            actual_moves.append(act)
            if pred == 'HIKE':
                real_sig_savings += act
            elif pred == 'DROP':
                real_sig_savings += -act
                
        n_perm = 1000
        perm_savings = []
        for _ in range(n_perm):
            shuffled_actuals = np.random.permutation(actual_moves)
            sh_sav = 0.0
            for p_dir, sh_act in zip(pred_dirs, shuffled_actuals):
                if p_dir == 'HIKE':
                    sh_sav += sh_act
                elif p_dir == 'DROP':
                    sh_sav += -sh_act
            perm_savings.append(sh_sav)
            
        better_trials = np.sum(np.array(perm_savings) >= real_sig_savings)
        p_value = float((better_trials + 1) / (n_perm + 1))
        
        if p_value < 0.05:
            if p_value < 0.001:
                p_value_str = "p < 0.001"
                p_value_note = "The model shows a highly significant trading edge (p < 0.001). This means there is less than a 0.1% probability that these savings were achieved by random chance."
            else:
                p_value_str = f"p = {p_value:.3f}"
                p_value_note = f"The model shows a statistically significant trading edge (p = {p_value:.3f} < 0.05). This means there is less than a 5% probability that these savings were achieved by random chance."
        else:
            p_value_str = f"p = {p_value:.3f}"
            p_value_note = f"The model significance is within normal bounds (p = {p_value:.3f} >= 0.05). This suggests that recent price moves contain more noise; continue to monitor performance parameters."

    # Plotting (Cropped to last 90 days for clarity)
    plot_cutoff = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(days=90)
    df_plot = df[df['timestamp_dt'] >= plot_cutoff].copy()
    if len(df_plot) < 10:
        df_plot = df.tail(60).copy()
        
    df_plot = df_plot.sort_values('timestamp_dt').copy()
    df_plot['plot_savings'] = df_plot['savings_cents'].cumsum()
    
    df_alerts_plot = df_alerts[df_alerts['timestamp_dt'] >= df_plot['timestamp_dt'].min()].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#f8fafc')
    ax2.set_facecolor('#f8fafc')
    
    # Plot recent cumulative savings
    plot_dates = [d.strftime("%Y-%m-%d") for d in df_plot['timestamp_dt']]
    ax1.plot(plot_dates, df_plot['plot_savings'], marker='o', markersize=4, label='Cumulative Savings (¢/gal)', color='#22c55e', linewidth=2.5)
    
    # Rolling trend (rolling 5 active alerts)
    df_alerts_window = df_plot[df_plot['predicted_direction'].isin(['HIKE', 'DROP'])].copy()
    if len(df_alerts_window) >= 5:
        trend = df_alerts_window['plot_savings'].rolling(window=5, min_periods=1).mean()
        trend_dates = [d.strftime("%Y-%m-%d") for d in df_alerts_window['timestamp_dt']]
        ax1.plot(trend_dates, trend, linestyle='--', color='#3b82f6', linewidth=2, label='Rolling Trend (5 alerts)')

    ax1.axhline(0, color='#94a3b8', linestyle='-', linewidth=1.2)
    ax1.set_title('Expected Savings in Last 90 Days (¢/gal)', fontsize=13, fontweight='bold', color='#1e293b', pad=10)
    ax1.set_ylabel('Cents per Gallon Saved', fontsize=10, color='#475569')
    
    if len(plot_dates) > 10:
        xticks_indices = np.linspace(0, len(plot_dates) - 1, 10, dtype=int)
        ax1.set_xticks(xticks_indices)
        ax1.set_xticklabels([plot_dates[i] for i in xticks_indices], rotation=15)
    else:
        ax1.set_xticks(range(len(plot_dates)))
        ax1.set_xticklabels(plot_dates, rotation=15)
        
    ax1.tick_params(colors='#64748b', labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#e2e8f0')
    ax1.grid(color='#e2e8f0', linestyle='--', alpha=0.7)
    ax1.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9)
    
    # Bottom Subplot: Rolling 90-Day Alert Precision (%)
    if has_90d_history and len(df_alerts_plot) > 0:
        alert_plot_dates = [d.strftime("%Y-%m-%d") for d in df_alerts_plot['timestamp_dt']]
        ax2.plot(alert_plot_dates, df_alerts_plot['rolling_precision'], marker='s', markersize=4, label='90-Day Rolling Precision', color='#8b5cf6', linewidth=2.5)
        ax2.axhline(53, color='#ef4444', linestyle=':', linewidth=1.5, label='RBOB Floor (53%)')
        ax2.axhline(60, color='#f97316', linestyle=':', linewidth=1.5, label='HO Floor (60%)')
        ax2.set_ylabel('Precision (%)', fontsize=10, color='#475569')
        ax2.set_ylim(0, 105)
        
        if len(alert_plot_dates) > 10:
            xticks_indices_2 = np.linspace(0, len(alert_plot_dates) - 1, 10, dtype=int)
            ax2.set_xticks(xticks_indices_2)
            ax2.set_xticklabels([alert_plot_dates[i] for i in xticks_indices_2], rotation=15)
        else:
            ax2.set_xticks(range(len(alert_plot_dates)))
            ax2.set_xticklabels(alert_plot_dates, rotation=15)
            
        ax2.tick_params(colors='#64748b', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_color('#e2e8f0')
        ax2.grid(color='#e2e8f0', linestyle='--', alpha=0.7)
        ax2.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Insufficient history for rolling precision\n(accumulating 90 days of prediction history)",
                 ha='center', va='center', fontsize=11, color='#64748b', transform=ax2.transAxes)
        ax2.set_ylim(0, 100)
        ax2.set_xlim(0, 1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_color('#e2e8f0')
            
    ax2.set_title('90-Day Rolling Alert Precision (%)', fontsize=13, fontweight='bold', color='#1e293b', pad=10)
    ax2.set_xlabel('Alert Date', fontsize=10, color='#475569')
    
    plt.tight_layout()
    
    report_date = datetime.now(TZ).strftime("%Y-%m-%d")
    chart_filename = f"report_{report_date}.png"
    chart_path = os.path.join(REPORTS_DIR, chart_filename)
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    # Send Email
    email_user = os.environ.get('GMAIL_USER')
    email_pass = os.environ.get('GMAIL_APP_PASSWORD')
    email_to_env = os.environ.get('TO_EMAIL', '')

    if not email_user or not email_pass:
        print("Missing email credentials. Cannot send report.")
        return

    recipients = [e.strip() for e in email_to_env.split(',') if e.strip()]
    
    # Deduplicate keeping order
    seen = set()
    emails = []
    for r in recipients:
        if r not in seen:
            seen.add(r)
            emails.append(r)
            
    if not emails:
        emails = [email_user]
        
    email_to = ", ".join(emails)
        
    subject = f"Weekly Performance Report - {report_date}"
    
    # HTML template structured for email client compatibility (table-based, inline CSS, no glassmorphism)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 20px;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f1f5f9;">
            <tr>
                <td align="center" style="padding: 20px 0;">
                    <table width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; border-collapse: separate;">
                        <!-- Header -->
                        <tr>
                            <td style="background-color: #f8fafc; padding: 24px; text-align: center; border-bottom: 1px solid #e2e8f0;">
                                <h1 style="margin: 0; color: #0f172a; font-size: 24px; font-weight: 600; line-height: 1.2;">Weekly Performance Report</h1>
                                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 14px; font-weight: 500;">Graves Oil Predictive Engine</p>
                            </td>
                        </tr>
                        
                        <!-- Content -->
                        <tr>
                            <td style="padding: 32px 24px;">
                                
                                <!-- WEEKLY ACTIVITY SECTION -->
                                <h2 style="color: #0f172a; font-size: 18px; margin: 0 0 16px 0; font-weight: 600; border-bottom: 2px solid #3b82f6; padding-bottom: 6px;">This Week's Activity (Last 7 Days)</h2>
                                <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 24px;">
                                    <tr>
                                        <td width="31%" align="center" style="background-color: #f8fafc; padding: 14px 8px; border-radius: 6px; border: 1px solid #e2e8f0;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Alerts Fired</div>
                                            <div style="color: #0f172a; font-size: 20px; font-weight: 700; margin-top: 6px;">{week_active_fired}</div>
                                        </td>
                                        <td width="3%"></td>
                                        <td width="32%" align="center" style="background-color: #f8fafc; padding: 14px 8px; border-radius: 6px; border: 1px solid #e2e8f0;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Weekly Precision</div>
                                            <div style="color: #3b82f6; font-size: 20px; font-weight: 700; margin-top: 6px;">{week_precision:.1f}%</div>
                                        </td>
                                        <td width="3%"></td>
                                        <td width="31%" align="center" style="background-color: #f8fafc; padding: 14px 8px; border-radius: 6px; border: 1px solid #e2e8f0;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Weekly Savings</div>
                                            <div style="color: #22c55e; font-size: 20px; font-weight: 700; margin-top: 6px;">{week_savings_cents:+.2f}¢</div>
                                            <div style="color: #16a34a; font-size: 11px; font-weight: 600; margin-top: 2px;">(${week_savings_dollars:,.2f} Max Modeled)*</div>
                                            <div style="color: #10b981; font-size: 10px; font-style: italic; margin-top: 2px;">~${est_realized_week_dollars:,.2f} Est. Realized ({dispatch_rate * 100:.0f}% dispatch rate)</div>
                                        </td>
                                    </tr>
                                </table>

                                <!-- RECENT ALERTS LOG TABLE -->
                                <h3 style="color: #334155; font-size: 14px; margin: 20px 0 8px 0; font-weight: 600;">Recent Resolved Alerts Log</h3>
                                <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 28px; border-collapse: collapse;">
                                    <thead>
                                        <tr style="background-color: #f8fafc; border-bottom: 2px solid #e2e8f0; text-align: left;">
                                            <th style="padding: 8px; color: #475569; font-size: 11px; font-weight: 600; text-transform: uppercase;">Date</th>
                                            <th style="padding: 8px; color: #475569; font-size: 11px; font-weight: 600; text-transform: uppercase;">Commodity</th>
                                            <th style="padding: 8px; color: #475569; font-size: 11px; font-weight: 600; text-transform: uppercase;">Signal</th>
                                            <th style="padding: 8px; color: #475569; font-size: 11px; font-weight: 600; text-transform: uppercase;">Actual Move</th>
                                            <th style="padding: 8px; color: #475569; font-size: 11px; font-weight: 600; text-transform: uppercase;">Result</th>
                                            <th style="padding: 8px; color: #475569; font-size: 11px; font-weight: 600; text-transform: uppercase; text-align: right;">Savings</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {recent_alerts_html}
                                    </tbody>
                                </table>

                                <!-- LIFETIME PERFORMANCE SECTION -->
                                <h2 style="color: #0f172a; font-size: 18px; margin: 24px 0 16px 0; font-weight: 600; border-bottom: 2px solid #10b981; padding-bottom: 6px;">Lifetime Model Calibration Performance</h2>
                                <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 12px;">
                                    <tr>
                                        <td width="48%" align="center" style="background-color: #f8fafc; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px; display: inline-block; vertical-align: top; width: 44%;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Alerts Fired</div>
                                            <div style="color: #0f172a; font-size: 22px; font-weight: 700; margin-top: 6px;">{total_active_alerts}</div>
                                            <div style="color: #64748b; font-size: 11px; margin-top: 2px;">active alerts (excl. flat)</div>
                                        </td>
                                        <td width="4%"></td>
                                        <td width="48%" align="center" style="background-color: #f8fafc; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px; display: inline-block; vertical-align: top; width: 44%;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Model Precision</div>
                                            <div style="color: #10b981; font-size: 22px; font-weight: 700; margin-top: 6px;">{lifetime_precision:.1f}%</div>
                                            <div style="color: #64748b; font-size: 11px; margin-top: 2px;">correct recommendations</div>
                                        </td>
                                    </tr>
                                    <tr style="height: 12px;"><td></td></tr>
                                    <tr>
                                        <td width="48%" align="center" style="background-color: #f8fafc; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0; display: inline-block; vertical-align: top; width: 44%;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Avg Savings / Active Alert</div>
                                            <div style="color: #10b981; font-size: 22px; font-weight: 700; margin-top: 6px;">{avg_savings_per_active_alert_cents:+.2f}¢/gal</div>
                                            <div style="color: #16a34a; font-size: 11px; font-weight: 600; margin-top: 2px;">(${avg_savings_per_truck_dollars:,.2f} per truck)*</div>
                                        </td>
                                        <td width="4%"></td>
                                        <td width="48%" align="center" style="background-color: #f8fafc; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0; display: inline-block; vertical-align: top; width: 44%;">
                                            <div style="color: #64748b; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Hypothetical Cum. Savings</div>
                                            <div style="color: #22c55e; font-size: 22px; font-weight: 700; margin-top: 6px;">{lifetime_savings_cents:+.2f}¢/gal</div>
                                            <div style="color: #16a34a; font-size: 11px; font-weight: 600; margin-top: 2px;">(${lifetime_savings_dollars:,.2f} Max Modeled)*</div>
                                            <div style="color: #10b981; font-size: 10px; font-style: italic; margin-top: 2px;">~${est_realized_lifetime_dollars:,.2f} Est. Realized ({dispatch_rate * 100:.0f}% dispatch rate)</div>
                                        </td>
                                    </tr>
                                </table>

                                <!-- EXECUTION DISCLAIMER & BENCHMARK NOTE -->
                                <table width="100%" cellpadding="12" cellspacing="0" border="0" style="background-color: #fffbeb; border-left: 4px solid #f59e0b; border-right: 1px solid #fef3c7; border-top: 1px solid #fef3c7; border-bottom: 1px solid #fef3c7; border-radius: 4px; margin-bottom: 28px;">
                                    <tr>
                                        <td>
                                            <p style="margin: 0; font-size: 12px; color: #b45309; font-weight: bold; text-transform: uppercase; letter-spacing: 0.03em;">
                                                * Execution Benchmark Disclaimer
                                            </p>
                                            <p style="margin: 6px 0 0 0; font-size: 11px; color: #b45309; line-height: 1.4;">
                                                Because the system has no visibility into your physical tank levels or carrier dispatch delays, savings calculations are initial theoretical limits. Under your load-time price lock contract with Graves Oil, orders must physically load before midnight CT to secure same-day pricing.
                                                <strong>To find your estimated realized savings:</strong> We discount modeled savings by {100 - dispatch_rate * 100:.0f}% (based on a user-configured dispatch same-day load rate of {dispatch_rate * 100:.0f}%) to account for dispatch delays and physical capacity constraints. Or, multiply your actual delivery volume by the <strong>Average Savings per Active Alert ({avg_savings_per_active_alert_cents:+.2f}&cent;/gal)</strong>.
                                            </p>
                                        </td>
                                    </tr>
                                </table>

                                <!-- Significance -->
                                <h3 style="color: #334155; font-size: 16px; margin: 24px 0 12px 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; font-weight: 600;">Model Significance ({sig_period_str})</h3>
                                <table width="100%" cellpadding="14" cellspacing="0" border="0" style="background-color: #f8fafc; border-left: 4px solid #3b82f6; border-right: 1px solid #e2e8f0; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; border-radius: 4px; margin-bottom: 28px;">
                                    <tr>
                                        <td>
                                            <p style="margin: 0; font-size: 13px; color: #0f172a; font-weight: 700;">
                                                Permutation P-Value: {p_value_str}
                                            </p>
                                            <p style="margin: 6px 0 0 0; font-size: 12px; color: #475569; line-height: 1.5;">
                                                {p_value_note}
                                            </p>
                                        </td>
                                    </tr>
                                </table>

                                <!-- System Health & Stability Checks -->
                                <h3 style="color: #334155; font-size: 16px; margin: 24px 0 12px 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; font-weight: 600;">System Health &amp; Stability Checks</h3>
                                <table width="100%" cellpadding="14" cellspacing="0" border="0" style="background-color: #f8fafc; border-left: 4px solid #10b981; border-right: 1px solid #e2e8f0; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; border-radius: 4px; margin-bottom: 28px;">
                                    <tr>
                                        <td>
                                            <p style="margin: 0; font-size: 13px; color: #0f172a; font-weight: 700;">
                                                Alert Frequency Monitoring
                                            </p>
                                            <p style="margin: 4px 0 12px 0; font-size: 12px; color: #475569; line-height: 1.5;">
                                                {freq_note}
                                            </p>
                                            <p style="margin: 12px 0 0 0; font-size: 13px; color: #0f172a; font-weight: 700;">
                                                Basis Stability (Mann-Kendall 90-Day Trend Check)
                                            </p>
                                            <p style="margin: 4px 0 0 0; font-size: 12px; color: #475569; line-height: 1.5;">
                                                <strong>Gas (RBOB):</strong> {basis_results.get('RB', {}).get('note', 'N/A')}<br>
                                                <strong>Diesel (HO):</strong> {basis_results.get('HO', {}).get('note', 'N/A')}
                                            </p>
                                            { f'<p style="margin: 10px 0 0 0; font-size: 12px; color: #ef4444; font-weight: bold; line-height: 1.5;">{basis_warning}</p>' if basis_warning else '' }
                                        </td>
                                    </tr>
                                </table>

                                <!-- Confusion Matrix -->
                                <h3 style="color: #334155; font-size: 16px; margin: 0 0 16px 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; font-weight: 600;">Prediction Confusion Matrix (Lifetime)</h3>
                                <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 32px; font-size: 14px;">
                                    <tr style="border-bottom: 1px solid #f1f5f9;">
                                        <td style="padding: 12px 0; color: #475569;">Correct Hikes Predicted <span style="color: #94a3b8; font-size: 12px;">(Saved money)</span></td>
                                        <td style="padding: 12px 0; text-align: right; font-weight: 600; color: #22c55e;">{correct_hikes}</td>
                                    </tr>
                                    <tr style="border-bottom: 1px solid #f1f5f9;">
                                        <td style="padding: 12px 0; color: #475569;">Correct Drops Predicted <span style="color: #94a3b8; font-size: 12px;">(Avoided loss)</span></td>
                                        <td style="padding: 12px 0; text-align: right; font-weight: 600; color: #22c55e;">{correct_drops}</td>
                                    </tr>
                                    <tr style="border-bottom: 1px solid #f1f5f9;">
                                        <td style="padding: 12px 0; color: #475569;">False Alarms <span style="color: #94a3b8; font-size: 12px;">(Flat Next Day &mdash; No loss)</span></td>
                                        <td style="padding: 12px 0; text-align: right; font-weight: 600; color: #f59e0b;">{false_flat}</td>
                                    </tr>
                                    <tr style="border-bottom: 1px solid #f1f5f9;">
                                        <td style="padding: 12px 0; color: #475569;">False Alarms <span style="color: #ef4444; font-size: 12px;">(Wrong Direction &mdash; Real loss)</span></td>
                                        <td style="padding: 12px 0; text-align: right; font-weight: 600; color: #ef4444;">{false_wrong_dir}</td>
                                    </tr>
                                    <tr style="border-bottom: 1px solid #f1f5f9;">
                                        <td style="padding: 12px 0; color: #475569;">Missed Moves <span style="color: #94a3b8; font-size: 12px;">(Predicted Flat)</span></td>
                                        <td style="padding: 12px 0; text-align: right; font-weight: 600; color: #64748b;">{missed_moves}</td>
                                    </tr>
                                </table>

                                <!-- Chart -->
                                <h3 style="color: #334155; font-size: 16px; margin: 0 0 16px 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; font-weight: 600;">Recent Performance Trends (Last 90 Days)</h3>
                                <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                    <tr>
                                        <td align="center" style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
                                            <img src="cid:chart_img" style="width: 100%; max-width: 550px; height: auto; display: block;" alt="Recent Savings and Precision Chart" />
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Footer -->
                        <tr>
                            <td style="background-color: #f8fafc; padding: 16px; text-align: center; border-top: 1px solid #e2e8f0;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Automated by Graves Oil Pricing Predictor</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = email_user
    msg['To'] = email_to
    
    alt = MIMEMultipart('alternative')
    alt.attach(MIMEText(html, 'html'))
    msg.attach(alt)
    
    with open(chart_path, 'rb') as f:
        img = MIMEImage(f.read(), 'png')
        img.add_header('Content-ID', '<chart_img>')
        msg.attach(img)
        
    emails = [e.strip() for e in email_to.split(',') if e.strip()]
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=30)
        server.starttls()
        server.login(email_user, email_pass)
        server.sendmail(email_user, emails, msg.as_string())
        server.quit()
        print("Weekly report email sent.")
    except Exception as e:
        print(f"Failed to send email: {mask_sensitive_text(e)}")

if __name__ == "__main__":
    main()
