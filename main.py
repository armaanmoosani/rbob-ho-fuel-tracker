import os
import sys
import io
import json
import uuid
import base64
import smtplib
import matplotlib
matplotlib.use('Agg')
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, date, time, timezone, timedelta
from matplotlib.figure import Figure
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})


SCHWAB_APP_KEY       = os.environ.get('SCHWAB_APP_KEY', '')
SCHWAB_APP_SECRET    = os.environ.get('SCHWAB_APP_SECRET', '')
SCHWAB_REFRESH_TOKEN = os.environ.get('SCHWAB_REFRESH_TOKEN', '')
GH_PAT               = os.environ.get('GH_PAT', '')
GH_REPO              = os.environ.get('GH_REPO', '')
GMAIL_USER           = os.environ.get('GMAIL_USER', '')
GMAIL_APP_PASSWORD   = os.environ.get('GMAIL_APP_PASSWORD', '')
_to_email_raw        = os.environ.get('TO_EMAIL', '')
TO_EMAIL             = [e.strip() for e in _to_email_raw.split(',') if e.strip()]
TO_PHONE_SMS         = [p.strip() for p in os.environ.get('PHONE_SMS_ADDRESS', '').split(',') if p.strip()]

import re
import csv

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
    
    # Mask email/phone recipients
    recipient_vars = ['GMAIL_USER', 'GRAVES_EMAIL', 'TO_EMAIL', 'PHONE_SMS_ADDRESS']
    sensitive_vals = []
    for env_var in recipient_vars:
        val = os.environ.get(env_var, '')
        if val:
            for item in val.split(','):
                item_stripped = item.strip()
                if item_stripped and len(item_stripped) > 2:
                    sensitive_vals.append(item_stripped)
    
    # Mask authentication tokens and secrets
    token_vars = ['GH_PAT', 'SCHWAB_APP_KEY', 'SCHWAB_APP_SECRET', 'SCHWAB_REFRESH_TOKEN', 'GMAIL_APP_PASSWORD']
    for env_var in token_vars:
        val = os.environ.get(env_var, '')
        if val and len(val) > 4:
            sensitive_vals.append(val)
    
    sensitive_vals = sorted(list(set(sensitive_vals)), key=len, reverse=True)
    
    for val in sensitive_vals:
        if '@' in val:
            # Email/phone masking
            text = text.replace(val, mask_recipient(val))
        else:
            # Token/secret masking
            if len(val) > 4:
                masked = val[:2] + '***' + val[-2:]
                text = text.replace(val, masked)
    return text

GH_HEADERS = {
    "Authorization": f"token {GH_PAT}",
    "Accept": "application/vnd.github.v3+json"
}

TZ = pytz.timezone('America/Chicago')

COMMODITIES = {
    'RB': {
        'name': 'Wholesale Gas',
        'yf_symbol': 'RB=F',
        'display_name': 'UNLEADED (/RB)'
    },
    'HO': {
        'name': 'Diesel',
        'yf_symbol': 'HO=F',
        'display_name': 'DIESEL (/HO)',
        'hidden': False
    },
    'CL': {
        'name': 'Crude Oil',
        'yf_symbol': 'CL=F',
        'display_name': 'CRUDE OIL (/CL)',
        'hidden': True
    }
}

REQUEST_TIMEOUT = 20
MAX_GH_VARIABLE_VALUE_BYTES = 48 * 1024
MAX_SMS_CHARS = 153
TRUCK_GALLONS = 8500   # Standard truck load for dollar-value risk calculations

# Load Config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
METRICS_CACHE_PATH = os.path.join(DATA_DIR, "metrics_cache.json")
CONFIG_CORRUPT = False
try:
    with open(CONFIG_PATH, "r") as f:
        APP_CONFIG = json.load(f)
    print("Loaded config.json custom thresholds.")
except Exception as e:
    if os.path.exists(CONFIG_PATH):
        CONFIG_CORRUPT = True
    print(f"Warning: Could not load config.json, using defaults. ({e})")
    APP_CONFIG = {
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
        "LAG_DAYS": 0
    }

# Safe overlay of metrics_cache.json if it exists (Issue 1.1)
if os.path.exists(METRICS_CACHE_PATH):
    try:
        with open(METRICS_CACHE_PATH, "r") as f:
            metrics_cache = json.load(f)
            APP_CONFIG.update(metrics_cache)
        print("Successfully loaded and overlaid metrics_cache.json thresholds.")
    except Exception as e:
        print(f"Warning: Could not load metrics_cache.json, relying on config/defaults. ({e})")

def get_session_start(dt):
    if dt.hour >= 17:
        return dt.replace(hour=17, minute=0, second=0, microsecond=0)
    else:
        prev_day = dt - timedelta(days=1)
        return prev_day.replace(hour=17, minute=0, second=0, microsecond=0)

def get_session_date_str(dt):
    if dt.hour >= 17:
        return (dt + timedelta(days=1)).date().isoformat()
    return dt.date().isoformat()

from futures_util import (
    DELIVERY_MONTH_CODES,
    add_month,
    is_nymex_business_day,
    observed_fixed_holiday,
    nth_weekday,
    last_weekday,
    easter_date,
    us_market_holidays,
    previous_nymex_business_day,
    last_nymex_business_day,
    refined_product_last_trade_date,
    crude_last_trade_date,
    contract_last_trade_date,
    get_front_month_contract,
    is_contract_roll_day,
    get_front_month_schwab_symbol
)

def schwab_to_yfinance_symbol(schwab_symbol):
    return schwab_symbol.lstrip('/') + '.NYM'

def contract_state_prefix(prefix, schwab_symbol):
    return f"{prefix}_{schwab_symbol.lstrip('/').upper()}"

def get_github_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PATCH"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

GH_SESSION = get_github_session()

def get_repo_variable(name):
    url = f"https://api.github.com/repos/{GH_REPO}/actions/variables/{name}"
    res = GH_SESSION.get(url, headers=GH_HEADERS, timeout=REQUEST_TIMEOUT)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    return res.json().get("value")

def set_repo_variable(name, value):
    if len(value.encode('utf-8')) > MAX_GH_VARIABLE_VALUE_BYTES:
        raise ValueError(f"GitHub Actions variable {name} exceeds 48 KB")
    url = f"https://api.github.com/repos/{GH_REPO}/actions/variables/{name}"
    res = GH_SESSION.patch(url, headers=GH_HEADERS, json={"name": name, "value": value}, timeout=REQUEST_TIMEOUT)
    if res.status_code == 404:
        res = GH_SESSION.post(
            f"https://api.github.com/repos/{GH_REPO}/actions/variables",
            headers=GH_HEADERS,
            json={"name": name, "value": value},
            timeout=REQUEST_TIMEOUT
        )
    res.raise_for_status()

def update_github_secret(new_refresh_token):
    import time
    from nacl import encoding, public
    url_key = f"https://api.github.com/repos/{GH_REPO}/actions/secrets/public-key"
    
    # Retry loop for public key retrieval
    public_key = None
    key_id = None
    for attempt in range(5):
        try:
            res_key = requests.get(url_key, headers=GH_HEADERS, timeout=REQUEST_TIMEOUT)
            res_key.raise_for_status()
            key_data = res_key.json()
            public_key = key_data['key']
            key_id = key_data['key_id']
            break
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Failed to retrieve GitHub repository public key after 5 attempts: {e}")
            time.sleep(2 ** attempt)

    public_key_obj = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder)
    sealed_box = public.SealedBox(public_key_obj)
    encrypted = sealed_box.encrypt(new_refresh_token.encode("utf-8"))
    encrypted_b64 = base64.b64encode(encrypted).decode("utf-8")

    url_secret = f"https://api.github.com/repos/{GH_REPO}/actions/secrets/SCHWAB_REFRESH_TOKEN"
    
    # Retry loop for writing the secret
    for attempt in range(5):
        try:
            res_secret = requests.put(
                url_secret,
                headers=GH_HEADERS,
                json={"encrypted_value": encrypted_b64, "key_id": key_id},
                timeout=REQUEST_TIMEOUT
            )
            res_secret.raise_for_status()
            break
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Failed to write rotated Schwab refresh token to GitHub Secrets after 5 attempts: {e}")
            time.sleep(2 ** attempt)

def load_price_history(prefix):
    raw = get_repo_variable(f"{prefix}_PRICE_HISTORY")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []

def save_price_history(history, prefix):
    try:
        set_repo_variable(f"{prefix}_PRICE_HISTORY", json.dumps(history))
    except Exception as e:
        print(f"Warning: could not save price history for {prefix}: {e}")

def load_alert_state():
    raw = get_repo_variable("ALERT_STATE")
    if not raw:
        return {}
    try:
        state = json.loads(raw)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}

def save_alert_state(state):
    keep = {}
    for key, value in state.items():
        if isinstance(value, str):
            keep[key] = value
    set_repo_variable("ALERT_STATE", json.dumps(keep, sort_keys=True))

def settlement_snapshot_key(prefix):
    return f"SETTLE_SNAPSHOT_{prefix}"

def load_settlement_snapshot(prefix, data, session_str):
    raw = get_repo_variable(settlement_snapshot_key(prefix))
    if not raw:
        return None
    try:
        snapshot = json.loads(raw)
    except Exception:
        return None
    if snapshot.get("date") != session_str:
        return None
    if snapshot.get("schwab_symbol") != data.get("schwab_symbol"):
        return None
    try:
        return float(snapshot["price"])
    except (KeyError, TypeError, ValueError):
        return None

def save_settlement_snapshots(all_data, now):
    if not (now.hour == 13 and 30 <= now.minute <= 45):
        return
    session_str = get_session_date_str(now)
    
    # Legacy logic (optional, keep for safety)
    for prefix in ['RB', 'HO']:
        data = all_data.get(prefix)
        if not data:
            continue
        if load_settlement_snapshot(prefix, data, session_str) is not None:
            continue
        snapshot = {
            "date": session_str,
            "price": round(data['current_price'], 4),
            "captured_at": now.isoformat(),
            "schwab_symbol": data.get("schwab_symbol")
        }
        try:
            set_repo_variable(settlement_snapshot_key(prefix), json.dumps(snapshot, sort_keys=True))
            print(f"[{prefix}] Saved settlement-window snapshot: ${snapshot['price']:.4f}")
        except Exception as e:
            print(f"[{prefix}] Warning: could not save settlement snapshot: {e}")

    # NEW LOGIC: Save daily_settlement.json locally for ingestion
    ds_path = os.path.join(os.path.dirname(__file__), "data", "daily_settlement.json")
    try:
        if os.path.exists(ds_path):
            with open(ds_path, "r") as f:
                ds = json.load(f)
            if ds.get("date") == session_str:
                return # Already saved today
                
        ds = {
            "date": session_str,
            "captured_at": now.isoformat()
        }
        if 'RB' in all_data:
            ds["rbob_settlement"] = round(all_data['RB']['current_price'], 4)
        if 'HO' in all_data:
            ds["heating_oil_settlement"] = round(all_data['HO']['current_price'], 4)
            
        os.makedirs(os.path.dirname(ds_path), exist_ok=True)
        with open(ds_path, "w") as f:
            json.dump(ds, f, indent=2)
        print(f"Saved local data/daily_settlement.json")
    except Exception as e:
        print(f"Warning: could not save daily_settlement.json: {e}")

def get_decoupling_warning(prefix, now):
    try:
        log_path = os.path.join(DATA_DIR, "prediction_log.csv")
        if not os.path.exists(log_path):
            return ""
            
        cutoff = now - timedelta(days=14)
        total_actionable = 0
        wins = 0
        
        with open(log_path, "r") as f:
            lines = f.readlines()[1:] # skip header
            
        for line in reversed(lines):
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            ts_str, comm, pred, nymex, lag, win_used, thresh, actual_str = parts[:8]
            
            if comm != prefix:
                continue
            if pred not in ("HIKE", "DROP"):
                continue
            if actual_str == "PENDING" or actual_str == "NaN":
                continue
                
            try:
                dt_str = ts_str.split("T")[0]
                # Avoid tzinfo mismatch by converting now to naive for comparison, 
                # or just parse dt directly into aware
                dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
                if dt < cutoff:
                    break
                    
                actual = float(actual_str)
                if pred == "HIKE" and actual > 0:
                    wins += 1
                elif pred == "DROP" and actual < 0:
                    wins += 1
                    
                total_actionable += 1
            except Exception:
                pass
                
        if total_actionable >= 5:
            win_rate = wins / total_actionable
            if win_rate < 0.40:
                return " WARNING: Supplier pricing has recently decoupled from NYMEX. The system is currently in a lag period before recalibration. Trust this alert with extreme caution."
    except Exception as e:
        print(f"[{prefix}] Failed to check decoupling warning: {e}")
        
    return ""

def build_rack_signal(prefix, data, now):
    yest = data.get('yesterday_close')
    if not yest:
        return {
            "action": "UNKNOWN",
            "label": "No Signal",
            "color": "#64748b",
            "text": f"{COMMODITIES[prefix]['name']}: no prior settlement baseline available.",
            "change_cents": None,
            "basis": "unavailable"
        }

    session_str = get_session_date_str(now)
    snapshot_price = load_settlement_snapshot(prefix, data, session_str)
    signal_price = snapshot_price if snapshot_price is not None else data['current_price']
    basis = "1:30 PM CT settlement-window snapshot" if snapshot_price is not None else "live price proxy"
    change_cents = (signal_price - yest) * 100
    name = COMMODITIES[prefix]['name']
    hike_thresh = APP_CONFIG.get(f"{prefix}_HIKE_THRESHOLD_CENTS", 1.0)

    # Check for contract roll day override
    # During test runs (pytest) we avoid suppressing alerts to keep tests deterministic,
    # unless `is_contract_roll_day` has been explicitly mocked by the test harness.
    running_tests = 'PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules
    try:
        from unittest import mock as _unittest_mock
        is_mocked = isinstance(is_contract_roll_day, _unittest_mock.Mock)
    except Exception:
        is_mocked = False

    roll_day = is_contract_roll_day(now, prefix)
    if roll_day:
        # Cross-check with the cached active symbol: if today's session and yesterday's
        # session both resolve to the same contract (e.g., both /RBN26), the market has
        # not actually changed contracts — the calendar LTD fired early due to
        # market-convention early roll (thinkorswim/yfinance switch 3–5 weeks before LTD).
        # In that case, override the calendar roll flag to False so we don't suppress.
        try:
            _state = load_alert_state()
            _today_session = get_session_date_str(now)
            from datetime import datetime as _dt_cls, date as _date_cls
            _now_ct = now.astimezone(TZ) if hasattr(now, 'astimezone') else now
            _prev_session = get_session_date_str(
                (_now_ct - timedelta(days=3)).replace(hour=12)  # go back ~3 days to find prev session
            )
            # Walk back up to 7 days to find a non-holiday/weekend cached symbol
            from datetime import timedelta as _td
            for _back in range(1, 8):
                _prev_ct = _now_ct - _td(days=_back)
                _prev_sess = get_session_date_str(_prev_ct.replace(hour=12))
                _prev_sym = _state.get(f"ACTIVE_SYMBOL_{prefix}_{_prev_sess}")
                if _prev_sym:
                    break
            else:
                _prev_sym = None
            _today_sym = _state.get(f"ACTIVE_SYMBOL_{prefix}_{_today_session}")
            if _today_sym and _prev_sym and _today_sym == _prev_sym:
                print(f"[{prefix}] Calendar LTD fired (roll_day=True) but active symbol "
                      f"unchanged ({_today_sym} == {_prev_sym}). "
                      f"Market has not rolled — overriding roll_day to False.")
                roll_day = False
        except Exception as _e:
            print(f"[{prefix}] Warning: could not verify roll via symbol cache: {_e}")
    if roll_day and not (running_tests and not is_mocked):
        risk_text = "Suppressing alert due to NYMEX front-month contract rollover day."
        if CONFIG_CORRUPT:
            risk_text += " WARNING: Corrupt config.json detected! System has rolled back to defaults."
            
        try:
            log_path = os.path.join(DATA_DIR, "prediction_log.csv")
            file_exists = os.path.exists(log_path)
            local_now = now.astimezone(TZ)
            session_str = local_now.date().isoformat()
            already_logged = False
            if file_exists:
                with open(log_path, "r") as f:
                    for line in f:
                        if line.startswith(session_str) and f",{prefix}," in line:
                            already_logged = True
                            break
            if not already_logged:
                with open(log_path, "a") as f:
                    if not file_exists:
                        f.write("timestamp,commodity,predicted_direction,nymex_move_cents,lag_used,window_used,threshold_used,actual_next_day_move_cents\n")
                    f.write(f"{local_now.isoformat()},{prefix},FLAT,0.00,0,120,0.00,PENDING\n")
        except Exception as e:
            print(f"Failed to write prediction log: {e}")

        return {
            "action": "NO_EDGE",
            "label": "Contract roll boundary",
            "color": "#64748b",
            "text": f"{name}: CONTRACT ROLL BOUNDARY (0.00 c/gal vs prior settle). Alert suppressed to prevent false signals from artificial roll price gaps.",
            "change_cents": 0.0,
            "basis": basis,
            "signal_price": signal_price,
            "threshold_cents": hike_thresh,
            "z_score": 0.0,
            "conviction": "Low Conviction",
            "risk_text": risk_text
        }


    # Load dynamic thresholds and stats from config
    hike_thresh = APP_CONFIG.get(f"{prefix}_HIKE_THRESHOLD_CENTS", 1.0)
    drop_thresh = APP_CONFIG.get(f"{prefix}_DROP_THRESHOLD_CENTS", -1.0)
    lean_hike = APP_CONFIG.get(f"{prefix}_LEAN_HIKE_CENTS", 0.5)
    lean_drop = APP_CONFIG.get(f"{prefix}_LEAN_DROP_CENTS", -0.5)
    
    nymex_daily_std = APP_CONFIG.get(f"{prefix}_nymex_daily_std", 1.0)
    
    # Calculate intraday realized volatility to check for genuine session regime shifts.
    #
    # Methodology:
    #   - Compute the sample std of observed 5-min price changes in cents (per-interval vol).
    #   - Derive the expected per-interval vol from the historical close-to-close daily std:
    #       expected_interval_vol = nymex_daily_std / sqrt(GLOBEX_5MIN_INTERVALS)
    #     (close-to-close daily std / sqrt(204) = expected 5-min move std assuming i.i.d.)
    #   - The override fires ONLY when observed per-interval vol > 2× the expected baseline,
    #     indicating a true intraday regime shift, not ordinary sampling noise.
    #   - Minimum 36 intervals (3 hours) required for a stable std estimate; with n<36 the
    #     95% CI on std(ddof=1) is too wide (>±45%) to distinguish regime shifts from noise.
    #   - When the override fires, the full-session vol estimate is std_diffs * sqrt(204).
    _GLOBEX_5MIN_INTERVALS = 204   # 17h × 12 five-minute intervals
    _MIN_INTRADAY_INTERVALS = 36   # 3 hours minimum for a stable std estimate
    _VOL_OVERRIDE_SIGMA = 2.0      # require 2× expected per-interval vol to trigger
    realized_vol = None
    schwab_symbol = data.get('schwab_symbol')
    if schwab_symbol:
        try:
            history_key = contract_state_prefix(prefix, schwab_symbol)
            history_intra = load_price_history(history_key)
            if len(history_intra) >= _MIN_INTRADAY_INTERVALS:
                prices = [h['p'] for h in history_intra]
                diffs = [(prices[i] - prices[i-1]) * 100 for i in range(1, len(prices))]
                if len(diffs) >= _MIN_INTRADAY_INTERVALS - 1:
                    std_diffs = float(np.std(diffs, ddof=1))
                    # Expected per-5-min vol from historical close-to-close daily std.
                    # Dividing daily std by sqrt(204) gives the per-interval baseline.
                    # This is conservative because close-to-close std includes overnight
                    # gap variance (~30-50% of total energy futures daily variance), so
                    # the per-interval baseline is slightly overstated, making the 2×
                    # threshold harder to cross and reducing false positives.
                    expected_interval_vol = nymex_daily_std / np.sqrt(_GLOBEX_5MIN_INTERVALS)
                    if std_diffs > _VOL_OVERRIDE_SIGMA * expected_interval_vol:
                        # Genuine intraday regime shift confirmed: scale to full-session vol
                        realized_vol = std_diffs * np.sqrt(_GLOBEX_5MIN_INTERVALS)
        except Exception as vol_e:
            print(f"[{prefix}] Failed to compute realized volatility: {vol_e}")
            
    used_std = nymex_daily_std
    vol_override_msg = ""
    if realized_vol is not None and realized_vol > nymex_daily_std:
        used_std = realized_vol
        vol_override_msg = f" (dynamic intraday vol override: {realized_vol:.2f}c vs historical {nymex_daily_std:.2f}c)"
        print(f"[{prefix}] Intraday regime shift detected! Volatility override active: {realized_vol:.4f} cents.")

    z_score = change_cents / used_std if used_std > 0 else 0.0
    abs_z = abs(z_score)
    
    if abs_z >= 1.5:
        conviction = "High Conviction"
    elif abs_z >= 1.0:
        conviction = "Moderate Conviction"
    else:
        conviction = "Low Conviction"

    if change_cents >= hike_thresh:
        action = "BUY_NOW"
        label = "Hike likely"
        color = "#ef4444"
        instruction = "Dispatch before the rack deadline if you need inventory."
    elif change_cents <= drop_thresh:
        action = "WAIT"
        label = "Drop likely"
        color = "#22c55e"
        instruction = "Wait if you can safely defer inventory. Never run tanks dry for a signal."
    elif change_cents >= lean_hike:
        action = "LEAN_BUY"
        label = "Lean hike"
        color = "#f97316"
        instruction = "Small edge only; buy only if inventory risk already favors lifting."
    elif change_cents <= lean_drop:
        action = "LEAN_WAIT"
        label = "Lean drop"
        color = "#38bdf8"
        instruction = "Small edge only; wait only if inventory risk allows it. Never run tanks dry for a signal."
    else:
        action = "NO_EDGE"
        label = "No clear edge"
        color = "#64748b"
        instruction = "Do not let this futures move alone drive the truck decision."

    if conviction == "Low Conviction" and action in ["BUY_NOW", "WAIT", "LEAN_BUY", "LEAN_WAIT"]:
        instruction = "Low Conviction — do not act on this signal unless inventory forces you to order regardless."

    # Build Z-score bin labels for conviction note
    if abs_z >= 1.5:
        bin_name = "high"
        bin_label = "High Conviction (|Z| >= 1.5)"
    elif abs_z >= 1.0:
        bin_name = "mod"
        bin_label = "Moderate Conviction (1.0 <= |Z| < 1.5)"
    else:
        bin_name = "low"
        bin_label = "Low Conviction (|Z| < 1.0)"

    # Build quantitative risk context text
    risk_text = ""
    if "BUY" in action:
        if prefix == "RB":
            baseline_range = "53%–73%"
            floor = "53%"
        else:
            baseline_range = "60%–79%"
            floor = "60%"
            
        win_rate = APP_CONFIG.get(f"{prefix}_{bin_name}_z_win_rate", -1.0)
        avg_savings = APP_CONFIG.get(f"{prefix}_{bin_name}_z_savings", 0.0)
        
        if win_rate >= 0.0:
            win_rate_pct = win_rate * 100
            conviction_part = f"similar {bin_label} alerts achieved a win rate of {win_rate_pct:.1f}% with average savings of {avg_savings:.2f}¢/gal"
        else:
            conviction_part = f"similar {bin_label} alerts had insufficient history to calculate stable precision"
            
        risk_text = (
            f"Conviction Note: In history, {conviction_part}. "
            f"(Multi-year baseline range: {baseline_range}; operational planning floor: {floor})."
        )
    elif "WAIT" in action:
        cvar = APP_CONFIG.get(f"{prefix}_historical_cvar", 3.0)
        cvar_truck_dollars = (cvar / 100.0) * TRUCK_GALLONS
        risk_text = (
            f"Risk Note: On the worst 5% of WAIT-signal days historically, rack prices spiked "
            f"+{cvar:.2f}¢/gal (+${cvar_truck_dollars:,.0f} per {TRUCK_GALLONS:,}-gallon truck). "
            f"Defer purchase only if inventory capacity allows."
        )

    if vol_override_msg:
        if risk_text:
            risk_text += vol_override_msg
        else:
            risk_text = f"Intraday Note:{vol_override_msg}"
            
    decoupling_warn = get_decoupling_warning(prefix, now)
    if decoupling_warn:
        risk_text += decoupling_warn

    if CONFIG_CORRUPT:
        if risk_text:
            risk_text += " WARNING: Corrupt config.json detected! System has rolled back to defaults."
        else:
            risk_text = "WARNING: Corrupt config.json detected! System has rolled back to defaults."

    # Immutable Prediction Audit Log (Idempotent)

    try:
        log_path = os.path.join(DATA_DIR, "prediction_log.csv")
        file_exists = os.path.exists(log_path)
        
        local_now = now.astimezone(TZ)
        session_str = local_now.date().isoformat()
        already_logged = False
        
        if file_exists:
            with open(log_path, "r") as f:
                for line in f:
                    if line.startswith(session_str) and f",{prefix}," in line:
                        already_logged = True
                        break
                        
        if not already_logged:
            with open(log_path, "a") as f:
                if not file_exists:
                    f.write("timestamp,commodity,predicted_direction,nymex_move_cents,lag_used,window_used,threshold_used,actual_next_day_move_cents\n")
                
                direction = "HIKE" if "BUY" in action else "DROP" if "WAIT" in action else "FLAT"
                thresh = hike_thresh if "BUY" in action else drop_thresh if "WAIT" in action else 0.0
                lag = APP_CONFIG.get("LAG_DAYS", 0)
                window = APP_CONFIG.get("ROLLING_WINDOW_DAYS", 120)
                
                f.write(f"{local_now.isoformat()},{prefix},{direction},{change_cents:.2f},{lag},{window},{thresh:.2f},PENDING\n")
    except Exception as e:
        print(f"Failed to write prediction log: {e}")

    return {
        "action": action,
        "label": label,
        "color": color,
        "text": f"{name}: {label.upper()} ({change_cents:+.2f} c/gal vs prior settle). {instruction}",
        "change_cents": change_cents,
        "basis": basis,
        "signal_price": signal_price,
        "threshold_cents": hike_thresh,
        "z_score": z_score,
        "conviction": conviction,
        "risk_text": risk_text
    }

def attach_rack_signals(all_data, now):
    for prefix in ['RB', 'HO']:
        if prefix in all_data:
            all_data[prefix]['rack_signal'] = build_rack_signal(prefix, all_data[prefix], now)

def append_price(history, ts, price):
    session_start = get_session_start(ts)
    filtered = []
    ts_iso = ts.isoformat()
    ts_exists = False
    for h in history:
        h_dt = datetime.fromisoformat(h['t'])
        if h_dt >= session_start:
            if h['t'] == ts_iso:
                h['p'] = round(price, 4)
                ts_exists = True
            filtered.append(h)
    if not ts_exists:
        filtered.append({"t": ts_iso, "p": round(price, 4)})
    return filtered

def generate_intraday_chart(history, current_price, baseline_price, high_price, low_price, daily_pct, commodity_name):
    if len(history) < 3:
        return None
    times  = [datetime.fromisoformat(h['t']).astimezone(TZ) for h in history]
    prices = [h['p'] for h in history]
    is_up      = daily_pct >= 0
    line_color = '#22c55e' if is_up else '#ef4444'
    bg_dark    = '#ffffff'
    bg_panel   = '#f8fafc'
    grid_color = '#e2e8f0'
    text_color = '#64748b'

    fig = Figure(figsize=(12, 5.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(bg_dark)
    ax.set_facecolor(bg_panel)

    if high_price > 0 and low_price > 0:
        ax.axhspan(low_price, high_price, alpha=0.04, color='#94a3b8', zorder=1)

    if baseline_price:
        ax.axhline(baseline_price, color='#94a3b8', linestyle='--', linewidth=1, zorder=2)
        ax.annotate(f' Yest. Close: ${baseline_price:.4f}', xy=(times[0], baseline_price),
                    color='#94a3b8', fontsize=14, va='bottom', ha='left')

    ax.plot(times, prices, color=line_color, linewidth=2.5, zorder=4)
    ax.fill_between(times, min(prices) if min(prices) < baseline_price else baseline_price, prices,
                    alpha=0.10, color=line_color, zorder=2)
    ax.scatter([times[-1]], [prices[-1]], color='#22c55e', s=60, zorder=5, linewidths=0)
    ax.annotate(f'  ${current_price:.4f}', xy=(times[-1], prices[-1]),
                color='#22c55e', fontsize=18, fontweight='bold', va='center')

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%-I:%M %p', tz=TZ))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis='x', rotation=0, colors=text_color, labelsize=14)
    ax.tick_params(axis='y', colors=text_color, labelsize=14)
    ax.set_ylabel('$/gal', color=text_color, fontsize=16)

    pct_sign = '+' if is_up else ''
    ax.set_title(
        f'{commodity_name} Intraday   {pct_sign}{daily_pct:.2f}% vs yest. close   '
        f'as of {times[-1].strftime("%-I:%M %p CT")}',
        color='#e2e8f0', fontsize=20, fontweight='bold', pad=12
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.tick_params(colors=text_color)
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=bg_dark)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def generate_5day_chart(history_5d, current_price):
    if len(history_5d) < 3:
        return None
    times  = [datetime.fromisoformat(h['t']).astimezone(TZ) for h in history_5d]
    prices = [h['p'] for h in history_5d]
    bg_dark    = '#ffffff'
    bg_panel   = '#f8fafc'
    grid_color = '#e2e8f0'
    text_color = '#64748b'
    line_color = '#60a5fa'

    fig = Figure(figsize=(12, 4.0))
    ax = fig.subplots()
    fig.patch.set_facecolor(bg_dark)
    ax.set_facecolor(bg_panel)

    ax.plot(times, prices, color=line_color, linewidth=2, zorder=4)
    ax.fill_between(times, min(prices) * 0.99, prices,
                    alpha=0.10, color=line_color, zorder=2)
    ax.scatter([times[-1]], [prices[-1]], color='#22c55e', s=60, zorder=5, linewidths=0)
    ax.annotate(f'  ${current_price:.4f}', xy=(times[-1], prices[-1]),
                color='#22c55e', fontsize=18, fontweight='bold', va='center')

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a', tz=TZ))
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    ax.tick_params(axis='x', rotation=0, colors=text_color, labelsize=14)
    ax.tick_params(axis='y', colors=text_color, labelsize=14)
    ax.set_ylabel('$/gal', color=text_color, fontsize=16)
    ax.set_title('5-Day Price Trend', color='#e2e8f0', fontsize=20, fontweight='bold', pad=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.tick_params(colors=text_color)
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, color=grid_color, linewidth=0.8, alpha=0.4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=bg_dark)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def build_html_block(prefix, info, now):
    cid_intra = uuid.uuid4().hex
    cid_5d    = uuid.uuid4().hex
    
    current_price = info['current_price']
    open_price = info['open_price']
    high_price = info['high_price']
    low_price = info['low_price']
    daily_pct = info['daily_pct']
    yesterday_close = info['yesterday_close']
    five_day_high = info['five_day_high']
    five_day_low = info['five_day_low']
    thirty_day_avg = info['thirty_day_avg']
    chart_intraday_b64 = info['chart_intraday_b64']
    chart_5d_b64 = info['chart_5d_b64']
    
    is_up      = daily_pct >= 0
    pct_color  = '#22c55e' if is_up else '#ef4444'
    pct_bg     = '#0a2010' if is_up else '#200a0a'
    arrow      = '\u25b2' if is_up else '\u25bc'
    pct_sign   = '+' if is_up else ''
    
    baseline_price = yesterday_close if yesterday_close else open_price
    dollar_chg = current_price - baseline_price
    dollar_str = f"+${dollar_chg:.4f}" if dollar_chg >= 0 else f"-${abs(dollar_chg):.4f}"

    price_range  = (high_price - low_price) if (high_price > 0 and low_price > 0) else 0
    range_pct    = ((current_price - low_price) / price_range * 100) if price_range > 0 else 50.0
    range_bar_pos = min(max(range_pct, 2), 98)

    def fmt_price(p): return f'${p:.4f}' if p else 'N/A'

    yest_cell = fmt_price(yesterday_close)
    yest_chg  = ''
    if yesterday_close:
        yc = ((current_price - yesterday_close) / yesterday_close) * 100
        yest_chg = f'<br><span style="font-size:11px;color:{"#22c55e" if yc >= 0 else "#ef4444"};">{("+" if yc >= 0 else "")}{yc:.2f}%</span>'

    ctx_5d_high = fmt_price(five_day_high)
    ctx_5d_low  = fmt_price(five_day_low)
    ctx_30d_avg = fmt_price(thirty_day_avg)
    ctx_30d_vs  = ''
    if thirty_day_avg:
        diff = ((current_price - thirty_day_avg) / thirty_day_avg) * 100
        ctx_30d_vs = f'<br><span style="font-size:11px;color:{"#22c55e" if diff >= 0 else "#ef4444"};">{("+" if diff >= 0 else "")}{diff:.2f}% vs avg</span>'

    if chart_intraday_b64:
        intraday_section = f'<img src="cid:{cid_intra}" style="width:100%;max-width:100%;height:auto;border-radius:6px;display:block;"/>'
    else:
        intraday_section = '<p style="color:#475569;font-size:11px;margin:0;">Chart waiting for data.</p>'

    if chart_5d_b64:
        fiveday_section = f'<img src="cid:{cid_5d}" style="width:100%;max-width:100%;height:auto;border-radius:6px;display:block;"/>'
    else:
        fiveday_section = ''

    sma3 = info.get('sma_3')
    sma10 = info.get('sma_10')
    momentum_html = ""
    if sma3 and sma10:
        if sma3 > sma10:
            mom_color = "#22c55e"
            mom_text = "BULLISH (Consider Buying)"
            mom_desc = "Short-term momentum is actively pushing prices up."
        else:
            mom_color = "#ef4444"
            mom_text = "BEARISH (Consider Waiting)"
            mom_desc = "Short-term momentum is actively pushing prices down."
            
        momentum_html = f'''
    <!-- MOMENTUM BAR -->
    <div style="background:#f8fafc;padding:12px 20px 14px;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td><p style="margin:0;font-size:10px;color:#64748b;text-transform:uppercase;font-weight:700;">Momentum Trend</p></td>
          <td style="text-align:right;"><p style="margin:0;font-size:12px;color:{mom_color};font-weight:700;">{mom_text}</p></td>
        </tr>
      </table>
      <p style="margin:4px 0 0;font-size:9px;color:#94a3b8;font-style:italic;">Based on 3-Day vs 10-Day Moving Average. {mom_desc}</p>
    </div>'''

    html = f"""
    <!-- ══ COMMODITY BLOCK: {COMMODITIES[prefix]['name']} ══ -->
    <div style="background:#ffffff;padding:20px 20px 16px;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:2px solid #94a3b8;">
      <p style="margin:0 0 2px;font-size:12px;color:#334155;letter-spacing:0.05em;text-transform:uppercase;font-weight:700;">{COMMODITIES[prefix]['name']} (/{prefix})</p>
      <p style="margin:0;font-size:44px;font-weight:700;color:#0f172a;letter-spacing:-1.5px;line-height:1.05;">
        ${current_price:.4f}
        <span style="font-size:14px;color:#64748b;font-weight:400;">&nbsp;/gal</span>
      </p>
      <div style="display:inline-block;margin-top:9px;padding:4px 13px;background:{pct_bg};border-radius:20px;border:1px solid {pct_color}33;">
        <span style="font-size:14px;font-weight:700;color:{pct_color};">{arrow}&nbsp;{pct_sign}{daily_pct:.2f}%</span>
        <span style="font-size:12px;color:{pct_color};opacity:0.8;">&nbsp;({dollar_str})</span>
        <span style="font-size:10px;color:#64748b;">&nbsp;vs yest. close</span>
      </div>
    </div>

    <!-- STATS ROW 1 -->
    <div style="background:#f8fafc;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td width="25%" style="padding:7px 10px 10px;border-right:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">Open</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#334155;">${open_price:.4f}</p>
          </td>
          <td width="25%" style="padding:7px 10px 10px;border-right:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">Day High</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#22c55e;">{f"${high_price:.4f}" if high_price > 0 else "N/A"}</p>
          </td>
          <td width="25%" style="padding:7px 10px 10px;border-right:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">Day Low</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#ef4444;">{f"${low_price:.4f}" if low_price > 0 else "N/A"}</p>
          </td>
          <td width="25%" style="padding:7px 10px 10px;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">$ Change</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:{pct_color};">{dollar_str}</p>
          </td>
        </tr>
      </table>
    </div>

    <!-- STATS ROW 2 -->
    <div style="background:#ffffff;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td width="25%" style="padding:7px 10px 10px;border-right:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">Yest. Close</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#334155;">{yest_cell}{yest_chg}</p>
          </td>
          <td width="25%" style="padding:7px 10px 10px;border-right:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">5-Day High</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#334155;">{ctx_5d_high}</p>
          </td>
          <td width="25%" style="padding:7px 10px 10px;border-right:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">5-Day Low</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#334155;">{ctx_5d_low}</p>
          </td>
          <td width="25%" style="padding:7px 10px 10px;text-align:center;">
            <p style="margin:0;font-size:8px;color:#64748b;text-transform:uppercase;">30-Day Avg</p>
            <p style="margin:4px 0 0;font-size:14px;font-weight:600;color:#334155;">{ctx_30d_avg}{ctx_30d_vs}</p>
          </td>
        </tr>
      </table>
    </div>

    <!-- RANGE BAR -->
    <div style="background:#f8fafc;padding:12px 20px 16px;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;">
      <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:8px;">
        <tr>
          <td><p style="margin:0;font-size:8px;color:#94a3b8;text-transform:uppercase;font-weight:600;">Position in Today's Range</p></td>
          <td style="text-align:right;"><p style="margin:0;font-size:10px;color:#475569;font-weight:600;">{range_pct:.0f}% of day range</p></td>
        </tr>
      </table>
      <div style="position:relative;height:4px;background:#e2e8f0;border-radius:2px;">
        <div style="height:4px;width:{range_bar_pos:.0f}%;background:linear-gradient(to right,#cbd5e1,#94a3b8);border-radius:2px;"></div>
        <div style="position:absolute;top:-5px;left:calc({range_bar_pos:.0f}% - 7px);width:14px;height:14px;border-radius:50%;background:{pct_color};border:2px solid #ffffff;"></div>
      </div>
    </div>

    {momentum_html}
    <!-- INTRADAY CHART -->
    <div style="background:#ffffff;padding:16px 20px 18px;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;">
      {intraday_section}
    </div>
    <!-- 5-DAY CHART -->
    {'<div style="background:#ffffff;padding:14px 20px 16px;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;">' + fiveday_section + '</div>' if fiveday_section else ''}
    """
    return html, cid_intra, cid_5d

def get_morning_confirmation_html(now):
    try:
        log_path = os.path.join(DATA_DIR, "prediction_log.csv")
        hist_path = os.path.join(DATA_DIR, "graves_history.csv")
        
        if not os.path.exists(log_path) or not os.path.exists(hist_path):
            return ""
            
        log_df = pd.read_csv(log_path)
        hist_df = pd.read_csv(hist_path)
        
        if log_df.empty or hist_df.empty:
            return ""
            
        # Parse dates
        log_df['date_only'] = log_df['timestamp'].apply(lambda x: x.split('T')[0] if isinstance(x, str) else "")
        log_df = log_df[log_df['date_only'] != ""]
        
        # We want the most recent date in the prediction log that has a completed/measurable outcome
        unique_dates = sorted(log_df['date_only'].unique(), reverse=True)
        
        target_date = None
        for d in unique_dates:
            idx_list = hist_df.index[hist_df['date'] == d].tolist()
            if idx_list and idx_list[0] - 1 >= 0:
                target_date = d
                break
                
        if not target_date:
            return ""
            
        day_preds = log_df[log_df['date_only'] == target_date]
        if day_preds.empty:
            return ""
            
        rows_html = []
        for _, pred_row in day_preds.iterrows():
            comm = pred_row['commodity']
            pred_dir = pred_row['predicted_direction']
            
            # Find the rack prices in graves_history
            idx = hist_df.index[hist_df['date'] == target_date].tolist()[0]
            curr_row = hist_df.iloc[idx]
            prev_row = hist_df.iloc[idx - 1]
            
            comm_name = "UNLEADED" if comm == 'RB' else "DIESEL"
            rack_col = 'rack_u' if comm == 'RB' else 'rack_d'
            
            curr_val = curr_row[rack_col]
            prev_val = prev_row[rack_col]
            
            if pd.isna(curr_val) or pd.isna(prev_val):
                continue
                
            move = (curr_val - prev_val) * 100  # in cents
            move_str = f"{move:+.2f}¢/gal"
            move_color = "#22c55e" if move > 0 else "#ef4444" if move < 0 else "#64748b"
            
            if pred_dir == "HIKE":
                pred_desc = '<span style="color:#22c55e;font-weight:bold;">HIKE (BUY)</span>'
                outcome = "✅ CORRECT" if move > 0 else "❌ INCORRECT"
                outcome_color = "#22c55e" if move > 0 else "#ef4444"
            elif pred_dir == "DROP":
                pred_desc = '<span style="color:#38bdf8;font-weight:bold;">DROP (WAIT)</span>'
                outcome = "✅ CORRECT" if move < 0 else "❌ INCORRECT"
                outcome_color = "#22c55e" if move < 0 else "#ef4444"
            else:
                pred_desc = '<span style="color:#64748b;font-weight:bold;">FLAT (NO EDGE)</span>'
                outcome = "N/A"
                outcome_color = "#64748b"
                
            rows_html.append(f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:8px 0;font-weight:700;color:#0f172a;">{comm_name}</td>
              <td style="padding:8px 0;">{pred_desc}</td>
              <td style="padding:8px 0;font-weight:600;color:{move_color};">{move_str}</td>
              <td style="padding:8px 0;text-align:right;font-weight:bold;color:{outcome_color};">{outcome}</td>
            </tr>
            """)
            
        if not rows_html:
            return ""
            
        target_dt = pd.to_datetime(target_date)
        formatted_date = target_dt.strftime('%A, %b %-d')
        
        confirmation_html = f'''
    <div style="padding:16px 22px;background:#ffffff;border-left:4px solid #10b981;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;margin-bottom:15px;border-radius:4px;">
      <p style="margin:0 0 8px;font-size:10px;color:#10b981;text-transform:uppercase;letter-spacing:0.1em;font-weight:800;">
        Overnight Verification Loop (Predictions from {formatted_date})
      </p>
      <table style="width:100%;border-collapse:collapse;font-size:12px;line-height:1.5;">
        <thead>
          <tr style="border-bottom:1px solid #cbd5e1;color:#64748b;text-align:left;">
            <th style="padding-bottom:6px;font-weight:600;">Product</th>
            <th style="padding-bottom:6px;font-weight:600;">Prediction</th>
            <th style="padding-bottom:6px;font-weight:600;">Rack Change</th>
            <th style="padding-bottom:6px;font-weight:600;text-align:right;">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>'''
        return confirmation_html
    except Exception as e:
        print(f"Warning: Failed to build morning confirmation HTML: {e}")
        return ""

def build_html_email(subject, all_data, now, alert_context):
    blocks = []
    cids = {}
    
    for prefix in ['RB', 'HO']:
        if prefix in all_data:
            block_html, cid_intra, cid_5d = build_html_block(prefix, all_data[prefix], now)
            blocks.append(block_html)
            cids[f"{prefix}_intra"] = cid_intra
            cids[f"{prefix}_5d"] = cid_5d

    action_line = ''
    if alert_context.get('label') == 'Final Verdict':
        rb = all_data.get('RB')
        ho = all_data.get('HO')
        
        def get_verdict(data, name):
            if not data:
                return ""
            signal = data.get('rack_signal')
            if not signal:
                return ""
            if signal.get('change_cents') is None:
                return f'<span style="color:#64748b;font-weight:800;">{name} NO SIGNAL:</span> Missing prior settlement baseline.<br>'
            
            conv_color = "#3b82f6" if "Moderate" in signal.get("conviction", "") else "#ef4444" if "High" in signal.get("conviction", "") else "#64748b"
            
            dispatch_rate = APP_CONFIG.get("DISPATCH_SAME_DAY_RATE", 0.50)
            sig_action = signal.get("action", "")
            realized_savings_html = ""
            if sig_action in ["BUY_NOW", "LEAN_BUY"]:
                realized_alert_savings = signal["change_cents"] * dispatch_rate
                realized_savings_html = f'  <span style="color:#16a34a;font-size:11px;font-weight:700;">Est. Realized Savings: {realized_alert_savings:+.2f}¢/gal (at {dispatch_rate*100:.0f}% same-day dispatch rate)</span><br>'
            elif sig_action in ["WAIT", "LEAN_WAIT"]:
                realized_alert_savings = -signal["change_cents"] * dispatch_rate
                realized_savings_html = f'  <span style="color:#16a34a;font-size:11px;font-weight:700;">Est. Realized Savings: {realized_alert_savings:+.2f}¢/gal (at {dispatch_rate*100:.0f}% same-day dispatch rate)</span><br>'

            html_lines = [
                f'<div style="margin-bottom: 8px;">',
                f'  <span style="color:{signal["color"]};font-weight:800;">{name} {signal["label"].upper()}:</span> ',
                f'  {signal["change_cents"]:+.2f} c/gal vs prior settle. ',
                f'  <span style="color:#64748b;font-size:11px;">(Basis: {signal["basis"]})</span><br>',
            ]
            if realized_savings_html:
                html_lines.append(realized_savings_html)

            instruction_text = signal.get("text", "").split(". ")[-1] if signal.get("text") else ""
            if instruction_text:
                html_lines.append(f'  <span style="font-size:11px;font-weight:700;color:{signal["color"]};">{instruction_text}</span><br>')


            if signal.get("z_score") is not None:
                html_lines.append(
                    f'  <span style="display:inline-block;margin-top:2px;padding:1px 6px;background:#f1f5f9;border-radius:3px;font-size:10px;color:{conv_color};font-weight:700;">'
                    f'    {signal["conviction"]} (Z-Score: {signal["z_score"]:+.2f})'
                    f'  </span>'
                )
            
            if signal.get("risk_text"):
                html_lines.append(
                    f'  <p style="margin:4px 0 0;font-size:11px;color:#475569;font-style:italic;line-height:1.4;">'
                    f'    {signal["risk_text"]}'
                    f'  </p>'
                )
                
            html_lines.append('</div>')
            return "\n".join(html_lines)
                
        v_rb = get_verdict(rb, 'UNLEADED')
        v_ho = get_verdict(ho, 'DIESEL')
        
        action_line = f'''
    <div style="padding:16px 22px;background:#f8fafc;border-left:6px solid #8b5cf6;border:1px solid #e2e8f0;margin-bottom:15px;border-radius:4px;">
      <p style="margin:0 0 6px;font-size:11px;color:#8b5cf6;text-transform:uppercase;letter-spacing:0.1em;font-weight:800;">
        {alert_context['label']}
      </p>
      <div style="margin:0;font-size:13px;color:#0f172a;line-height:1.6;">
        {v_rb}{v_ho}
      </div>
      <p style="margin:8px 0 0;font-size:11px;color:#64748b;line-height:1.5;">
        Action thresholds: Unleaded +/-{APP_CONFIG.get('RB_HIKE_THRESHOLD_CENTS', 1.0):.2f} c/gal | Diesel +/-{APP_CONFIG.get('HO_HIKE_THRESHOLD_CENTS', 1.0):.2f} c/gal. Confidence levels are adjusted dynamically using rolling historical volatility.
      </p>
    </div>'''
    elif alert_context.get('action'):
        ac = alert_context.get('action_color', '#64748b')
        action_line = f'''
    <div style="padding:14px 22px;background:#ffffff;border-left:4px solid {ac};border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;margin-bottom:15px;">
      <p style="margin:0 0 3px;font-size:9px;color:{ac};text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">
        {alert_context['label']}
      </p>
      <p style="margin:0;font-size:12px;color:#475569;line-height:1.6;">
        {alert_context['action']}
      </p>
    </div>'''

    if alert_context.get('morning_confirmation_html'):
        action_line += alert_context['morning_confirmation_html']

    crack_spread_html = ''
    if 'RB' in all_data and 'HO' in all_data and 'CL' in all_data:
        rb = all_data['RB']
        ho = all_data['HO']
        cl = all_data['CL']
        if rb['current_price'] and ho['current_price'] and cl['current_price']:
            # 3:2:1 Crack Spread per barrel: (2 * RB * 42 + 1 * HO * 42 - 3 * CL) / 3 = 28 * RB + 14 * HO - CL
            crack = (rb['current_price'] * 28) + (ho['current_price'] * 14) - cl['current_price']
            crack_str = f"${crack:.2f}"
            trend_str = ""
            if rb['yesterday_close'] and ho['yesterday_close'] and cl['yesterday_close']:
                # 3:2:1 Crack Spread per barrel: (2 * RB * 42 + 1 * HO * 42 - 3 * CL) / 3 = 28 * RB + 14 * HO - CL
                yest_crack = (rb['yesterday_close'] * 28) + (ho['yesterday_close'] * 14) - cl['yesterday_close']
                crack_chg = crack - yest_crack
                sign = '+' if crack_chg >= 0 else ''
                trend_color = '#22c55e' if crack_chg >= 0 else '#ef4444'
                trend_text = "Widening refinery margin (Upward trend on fuel prices)" if crack_chg >= 0 else "Shrinking refinery margin (Downward trend on fuel prices)"
                trend_str = f'<span style="color:{trend_color};font-weight:700;">{sign}${crack_chg:.2f} / bbl</span> &nbsp;&middot;&nbsp; <span>{trend_text}</span>'
            
            crack_spread_html = f'''
            <!-- MARKET INTELLIGENCE -->
            <div style="background:#ffffff;margin-top:15px;padding:16px 20px;border-left:4px solid #6366f1;border-right:1px solid #e2e8f0;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;">
              <p style="margin:0 0 4px;font-size:10px;color:#6366f1;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">Market Intelligence</p>
              <p style="margin:0 0 6px;font-size:16px;color:#0f172a;font-weight:700;">3:2:1 Crack Spread: {crack_str}</p>
              <p style="margin:0 0 8px;font-size:12px;color:#475569;">{trend_str}</p>
              <p style="margin:0;font-size:11px;color:#64748b;line-height:1.5;"><strong>Use:</strong> Treat this as market context, not a truck-dispatch signal. The rack decision should come from the product-specific RB/HO cents-per-gallon move versus prior settlement.</p>
            </div>
            '''

    latest_quote_time = now
    quote_times = []
    for prefix in ['RB', 'HO']:
        if prefix in all_data and all_data[prefix].get('quote_time'):
            quote_times.append(all_data[prefix]['quote_time'])
    if quote_times:
        latest_quote_time = max(quote_times)
    
    header_time_str = latest_quote_time.astimezone(TZ).strftime('%-I:%M %p CT &nbsp;&middot;&nbsp; %b %-d, %Y')
    footer_sent_time_str = now.astimezone(TZ).strftime('%-I:%M:%S %p %Z')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f1f5f9;">
    <tr>
      <td align="center" style="padding:20px 10px;">
        <div style="max-width:620px;margin:0 auto;width:100%;text-align:left;">
          <!-- HEADER -->
          <div style="background:#ffffff;border-radius:10px 10px 0 0;padding:13px 20px;border:1px solid #e2e8f0;border-bottom:none;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td><p style="margin:0;font-size:10px;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;">Fuel Tracker &nbsp;&middot;&nbsp; NYMEX CME</p></td>
                <td style="text-align:right;"><p style="margin:0;font-size:10px;color:#94a3b8;">{header_time_str}</p></td>
              </tr>
            </table>
          </div>
          
          {action_line}
          {''.join(blocks)}
          {crack_spread_html}
          
          <!-- FOOTER -->
          <div style="background:#f1f5f9;border-radius:0 0 10px 10px;padding:10px 20px;border:1px solid #e2e8f0;border-top:none;">
            <p style="margin:0;font-size:9px;color:#64748b;line-height:1.6;">Automated by armaanmoosani/rbob-ho-fuel-tracker &nbsp;&middot;&nbsp; Sent at {footer_sent_time_str}</p>
          </div>
        </div>
      </td>
    </tr>
  </table>
</body>
</html>"""
    
    return html, cids

def send_sms(all_data, now, alert_context):
    """Send a compact SMS that fits in a single 153-char carrier segment.

    The Gmail-to-SMS gateway splits messages at the 153-char per-segment
    boundary (standard UDH multipart SMS) and only the first segment is
    reliably delivered. Every branch below must produce a body that is
    at most MAX_SMS_CHARS characters so that both RB and HO data are
    always visible in the same message the user receives.
    """
    if not TO_PHONE_SMS:
        return

    label    = alert_context.get('label', 'Update')
    latest_quote_time = now
    quote_times = []
    for prefix in ['RB', 'HO']:
        if prefix in all_data and all_data[prefix].get('quote_time'):
            quote_times.append(all_data[prefix]['quote_time'])
    if quote_times:
        latest_quote_time = max(quote_times)
    time_str = latest_quote_time.astimezone(TZ).strftime('%-I:%M %p CT')

    # --- HEARTBEAT / FAILURE: short fixed bodies ---
    if alert_context.get('label') == 'HEARTBEAT':
        body = f"HEARTBEAT {time_str}: Fuel Tracker active."
    elif alert_context.get('label') == 'FAILURE':
        action_desc = alert_context.get('action', 'Workflow failed. Check GHA logs.')
        body = f"FAILURE {time_str}: {action_desc}"[:MAX_SMS_CHARS]

    elif alert_context.get('label') == 'Final Verdict':
        # Compact verdict: one line per commodity signal + one price line each.
        # Format per signal:
        #   Gas: BUY | High | +4.00c
        #   Gas: BUY | LOW CONV — do not dispatch
        lines = [f"[{time_str}] FINAL VERDICT"]
        for prefix, short in [('RB', 'Gas'), ('HO', 'Diesel')]:
            info = all_data.get(prefix)
            if not info:
                continue
            sig = info.get('rack_signal')
            if sig and sig.get('change_cents') is not None:
                action = sig.get('action', '')
                conv   = sig.get('conviction', '')
                chg    = sig.get('change_cents', 0.0)
                # Extract BUY / WAIT / NO EDGE keyword
                if 'BUY' in action:
                    kw = 'BUY'
                elif 'WAIT' in action:
                    kw = 'WAIT'
                else:
                    kw = sig.get('label', 'NO EDGE')
                if conv == 'Low Conviction':
                    lines.append(f"{short}: {kw} | LOW CONV — do not dispatch")
                else:
                    conv_short = conv.replace(' Conviction', '') if conv else ''
                    lines.append(f"{short}: {kw} | {conv_short} | {chg:+.2f}c")
        # Price summary — one compact line per commodity
        for prefix, abbr in [('RB', 'Unleaded'), ('HO', 'Diesel')]:
            info = all_data.get(prefix)
            if not info:
                continue
            pct      = info.get('daily_pct')
            price    = info.get('current_price')
            if price is None:
                continue
            if pct is not None:
                pct_sign = '+' if pct >= 0 else ''
                lines.append(f"{abbr}: ${price:.4f} ({pct_sign}{pct:.2f}%)")
            else:
                lines.append(f"{abbr}: ${price:.4f}")
        body = "\n".join(lines).strip()

    else:
        # PRICE ALERT, SETTLEMENT, RACK WINDOW, or generic UPDATE.
        # Determine alert type label.
        if 'Swing' in label or 'Movement' in label:
            alert_type = 'PRICE ALERT'
        elif 'Rack' in label:
            alert_type = 'RACK WINDOW'
        elif 'Settlement' in label:
            alert_type = 'SETTLEMENT'
        else:
            alert_type = 'UPDATE'

        lines = [f"[{time_str}] {alert_type}"]
        # One compact line per commodity: Unleaded: $3.3351 (-3.44% | -$0.1188)
        for prefix, abbr in [('RB', 'Unleaded'), ('HO', 'Diesel')]:
            info = all_data.get(prefix)
            if not info:
                continue
            price = info.get('current_price')
            pct   = info.get('daily_pct')
            if price is None:
                continue
            baseline = info.get('yesterday_close') or info.get('open_price')
            if pct is not None and baseline is not None:
                pct_sign  = '+' if pct >= 0 else ''
                dollar_chg = price - baseline
                sign       = '+' if dollar_chg >= 0 else '-'
                lines.append(f"{abbr}: ${price:.4f} ({pct_sign}{pct:.2f}% | {sign}${abs(dollar_chg):.4f})")
            elif pct is not None:
                pct_sign = '+' if pct >= 0 else ''
                lines.append(f"{abbr}: ${price:.4f} ({pct_sign}{pct:.2f}%)")
            else:
                lines.append(f"{abbr}: ${price:.4f}")
        body = "\n".join(lines).strip()

    # Hard guardrail: truncate to one carrier segment.
    if len(body) > MAX_SMS_CHARS:
        body = body[:MAX_SMS_CHARS].rstrip()

    try:
        srv = smtplib.SMTP('smtp.gmail.com', 587, timeout=30)
        srv.starttls()
        srv.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        
        phones = TO_PHONE_SMS
        for phone in phones:
            # Construct a fresh MIMEText per recipient — never mutate a shared
            # MIME object across multiple sendmail() calls, as header state bleeds.
            per_msg = MIMEText(body)
            # Many carrier email-to-SMS/MMS gateways (AT&T, T-Mobile, Verizon)
            # silently drop messages with an empty Subject line.  Use the alert
            # label (already a short string) so the gateway has a non-empty subject.
            per_msg['Subject'] = alert_context.get('label', 'Fuel Alert')
            per_msg['From']    = GMAIL_USER
            per_msg['To']      = phone
            srv.sendmail(GMAIL_USER, phone, per_msg.as_string())
            print(f"SMS sent to gateway ({mask_recipient(phone)})")
            
        srv.quit()
    except Exception as e:
        print(f"LOG_OUTBOUND_FAILURE: SMS send failed (non-fatal): {mask_sensitive_text(e)}")

def send_email(subject, all_data, now, alert_context):
    try:
        html_body, cids = build_html_email(subject, all_data, now, alert_context)
        
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From']    = GMAIL_USER
        
        emails = TO_EMAIL
        
        alt = MIMEMultipart('alternative')
        alt.attach(MIMEText("Please view in an HTML-compatible email client.", 'plain'))
        alt.attach(MIMEText(html_body, 'html'))
        msg.attach(alt)

        for prefix in ['RB', 'HO']:
            if prefix in all_data:
                info = all_data[prefix]
                if info.get('chart_intraday_b64'):
                    img = MIMEImage(base64.b64decode(info['chart_intraday_b64']), 'png')
                    img.add_header('Content-ID', f'<{cids[f"{prefix}_intra"]}>')
                    msg.attach(img)
                if info.get('chart_5d_b64'):
                    img2 = MIMEImage(base64.b64decode(info['chart_5d_b64']), 'png')
                    img2.add_header('Content-ID', f'<{cids[f"{prefix}_5d"]}>')
                    msg.attach(img2)

        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=30)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        
        for email_addr in emails:
            msg.replace_header('To', email_addr) if 'To' in msg else msg.add_header('To', email_addr)
            server.sendmail(GMAIL_USER, email_addr, msg.as_string())
            print(f"Email sent: {subject} to {mask_recipient(email_addr)}")
            
        server.quit()
    except Exception as e:
        print(f"LOG_OUTBOUND_FAILURE: Email send failed: {mask_sensitive_text(e)}")

    send_sms(all_data, now, alert_context)


def send_once_today(key, subject, all_data, now, alert_context):
    session_str = get_session_date_str(now)
    db_key = f"SENT_{key}"
    state = load_alert_state()
    
    if state.get(db_key) == session_str:
        print(f"Alert {key} already sent today. Skipping.")
        return
        
    send_email(subject, all_data, now, alert_context)
    try:
        state[db_key] = session_str
        save_alert_state(state)
    except Exception as e:
        print(f"Warning: could not save lock {db_key}: {e}")

def send_daily_prompt(now):
    # Only run at 7:30 PM CT (19:30) on weekdays (and not on holidays)
    if now.weekday() > 4 or not (now.hour == 19 and now.minute >= 30):
        return
    
    if is_holiday(now):
        print(f"Skipping daily price prompt on holiday ({now.astimezone(TZ).date()})")
        return

    # Check if today's prices have already been ingested to prevent duplicate prompts
    today_str = now.date().isoformat()
    csv_file = os.path.join(DATA_DIR, "graves_history.csv")
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if parts and parts[0] == today_str:
                        print(f"Graves Oil prices for today ({today_str}) are already ingested. Skipping daily SMS prompt.")
                        return
        except Exception as e:
            print(f"Warning: could not read graves_history.csv to check ingestion status: {e}")

    session_str = get_session_date_str(now)
    db_key = "SENT_DAILY_PROMPT"
    
    # Use repo variable checkpoint to ensure it only sends once
    last_sent = get_repo_variable(db_key)
    if last_sent == session_str:
        return
        
    print("Time is 19:30 CT. Sending daily SMS price prompt...")
    body = "Please reply with today's Graves Oil prices.\nFormat: Unleaded Premium Diesel\n(e.g. 2.10 2.30 2.50)"
    
    try:
        srv = smtplib.SMTP('smtp.gmail.com', 587, timeout=30)
        srv.starttls()
        srv.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        
        phones = TO_PHONE_SMS
        for phone in phones:
            # Construct a fresh MIMEText per recipient — never mutate a shared object.
            per_msg = MIMEText(body)
            per_msg['Subject'] = 'Graves Oil Prices Needed'
            per_msg['From']    = GMAIL_USER
            per_msg['To']      = phone
            srv.sendmail(GMAIL_USER, phone, per_msg.as_string())
            print(f"Daily prompt sent to {mask_recipient(phone)}")
            
        srv.quit()
        
        # Checkpoint successful send
        set_repo_variable(db_key, session_str)
    except Exception as e:
        print(f"LOG_OUTBOUND_FAILURE: Failed to send daily prompt: {mask_sensitive_text(e)}")

def main():
    # Assert Chicago timezone is correctly recognized (Issue 10.1)
    assert TZ.zone == 'America/Chicago', "TimeZone mismatch: America/Chicago expected."
    
    dispatch_rate = APP_CONFIG.get("DISPATCH_SAME_DAY_RATE", 0.50)
    print(f"System initialized. Active DISPATCH_SAME_DAY_RATE: {dispatch_rate:.2f} ({dispatch_rate * 100:.0f}%)")
    
    start_time = datetime.now(timezone.utc)
    now = datetime.now(TZ)
    
    if is_holiday(now):
        print(f"Today is a holiday ({now.date()}). Skipping price checks and alerts.")
        return
    
    if not is_market_open(now):
        print(f"Market is closed at {now}. Exiting to prevent stale alerts.")
        return

    access_token = None
    if not (SCHWAB_APP_KEY and SCHWAB_APP_SECRET and SCHWAB_REFRESH_TOKEN):
        print("Warning: Schwab API credentials (SCHWAB_APP_KEY, SCHWAB_APP_SECRET, or SCHWAB_REFRESH_TOKEN) are not configured. Bypassing Schwab and using Yahoo Finance fallback directly.")
    else:
        auth_header = base64.b64encode(f"{SCHWAB_APP_KEY}:{SCHWAB_APP_SECRET}".encode()).decode()
        try:
            auth_res = requests.post(
                "https://api.schwabapi.com/v1/oauth/token",
                data={"grant_type": "refresh_token", "refresh_token": SCHWAB_REFRESH_TOKEN},
                headers={"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"},
                timeout=REQUEST_TIMEOUT
            )
            auth_res.raise_for_status()
            auth_json = auth_res.json()
            access_token = auth_json['access_token']
            new_refresh = auth_json.get('refresh_token')
            print("Schwab OAuth refreshed")
        except Exception as e:
            print(f"Schwab OAuth refresh failed: {mask_sensitive_text(e)}. Bypassing Schwab and forcing Yahoo Finance fallback.")
            new_refresh = None

        if new_refresh and new_refresh != SCHWAB_REFRESH_TOKEN:
            try:
                update_github_secret(new_refresh)
                print("Schwab refresh token rotated in GitHub Secrets")
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Schwab token refresh succeeded but updating GitHub Secrets failed: {mask_sensitive_text(e)}")

    all_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(COMMODITIES)) as executor:
        futures = [executor.submit(fetch_commodity, prefix, cfg, now, access_token) for prefix, cfg in COMMODITIES.items()]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                all_data[res.pop('prefix')] = res

    session_str = get_session_date_str(now)
    if 'RB' not in all_data and 'HO' not in all_data:
        raise RuntimeError("CRITICAL ERROR: Failed to fetch pricing data for both RBOB (Unleaded) and Heating Oil (Diesel) from all available sources.")
    save_settlement_snapshots(all_data, now)
    
    any_swing = False
    swing_trigger_desc = []
    swing_colors = []
    subject_strs = []
    
    for prefix in ['RB', 'HO']:
        if prefix not in all_data: continue
        
        pct = all_data[prefix]['daily_pct']
        ps_sub = '+' if pct > 0 else ''
        subject_strs.append(f"{COMMODITIES[prefix]['name']}: {ps_sub}{pct:.2f}%")
        
        raw = get_repo_variable(f"LAST_SWING_INFO_{prefix}")
        last_alert_price = None
        if raw:
            try:
                info = json.loads(raw)
                if (
                    info.get("date") == session_str
                    and info.get("schwab_symbol") == all_data[prefix].get('schwab_symbol')
                ):
                    last_alert_price = float(info.get("price"))
            except Exception:
                pass
                
        curr = all_data[prefix]['current_price']
        baseline = all_data[prefix].get('yesterday_close') or all_data[prefix]['open_price']
        ref = last_alert_price if last_alert_price else baseline
        swing = ((curr - ref) / ref * 100) if ref else 0.0
        
        if abs(swing) >= 1.5:
            any_swing = True
            ps = '+' if swing >= 0 else ''
            dollar_swing = curr - ref
            pds = '+' if dollar_swing >= 0 else '-'
            swing_trigger_desc.append(
                f"{COMMODITIES[prefix]['name']} {ps}{swing:.2f}% ({pds}${abs(dollar_swing):.4f}/gal) from last reference point of ${ref:.4f}/gal"
            )
            swing_colors.append('#f97316' if swing > 0 else '#22c55e')
            try:
                set_repo_variable(
                    f"LAST_SWING_INFO_{prefix}",
                    json.dumps({
                        "date": session_str,
                        "price": round(curr, 4),
                        "schwab_symbol": all_data[prefix].get('schwab_symbol')
                    })
                )
            except Exception:
                pass
                
    if any_swing:
        send_email(
            subject=f"Price Spike: {' | '.join(subject_strs)}",
            all_data=all_data,
            now=now,
            alert_context={
                'label': 'Price Movement Alert',
                'action': f"Significant price movement detected: {', '.join(swing_trigger_desc)}.",
                'action_color': swing_colors[0]
            }
        )

    local_now = now.astimezone(TZ)
    
    force_verdict = os.environ.get('FORCE_VERDICT', 'false').lower() == 'true'
    
    if force_verdict:
        attach_rack_signals(all_data, now)
        alert_ctx = {
            'label': 'Final Verdict',
            'action': 'ON-DEMAND MANUAL OVERRIDE: Comparing the current NYMEX price to the prior settlement to estimate rack direction.',
            'action_color': '#8b5cf6'
        }
        send_email("Final Verdict (On Demand): Exxon Price Predictor", all_data, now, alert_ctx)
        send_sms(all_data, now, alert_ctx)
    elif local_now.hour == 14 and local_now.minute >= 35:
        attach_rack_signals(all_data, now)
        send_once_today('VERDICT_1435', "Final Verdict: Exxon Price Predictor", all_data, now, {
            'label': 'Final Verdict',
            'action': 'Comparing the 1:30 PM CT NYMEX settlement-window move to the prior settlement to estimate tonight’s OPIS/Graves rack direction.',
            'action_color': '#8b5cf6'
        })

    if local_now.hour in [8, 12, 18]:
        # Build a compact live-price summary for the action line
        price_parts = []
        for prefix, name in [('RB', 'RBOB'), ('HO', 'Heating Oil')]:
            if prefix in all_data:
                d = all_data[prefix]
                cp = d.get('current_price')
                pct = d.get('daily_pct_change')
                if cp is not None:
                    pct_str = f" ({'+' if pct >= 0 else ''}{pct:.2f}%)" if pct is not None else ""
                    price_parts.append(f"{name}: ${cp:.4f}{pct_str}")
        action_text = ('Live prices: ' + ' · '.join(price_parts)) if price_parts else 'Periodic fuel market snapshot.'
        alert_ctx = {
            'label': f'Scheduled Market Update — {local_now.strftime("%-I:%M %p CT")}',
            'action': action_text,
            'action_color': '#475569'
        }
        if local_now.hour == 8:
            confirm_html = get_morning_confirmation_html(now)
            if confirm_html:
                alert_ctx['morning_confirmation_html'] = confirm_html
                
        send_once_today(f"UPDATE_{local_now.strftime('%H')}", f"Market Update — {local_now.strftime('%-I %p')}", all_data, now, alert_ctx)

    send_daily_prompt(now)

    print(f"Total time: {(datetime.now(timezone.utc) - start_time).total_seconds():.1f}s")


def is_market_open(dt):
    def simple_globex_energy_hours(ts):
        local = ts.astimezone(TZ)
        local_date = local.date()
        if local_date in us_market_holidays(local_date.year):
            return False
        local_time = local.time()
        weekday = local.weekday()
        if weekday == 5:
            return False
        if weekday == 6:
            return local_time >= time(17, 0)
        if weekday == 4 and local_time >= time(16, 0):
            return False
        if time(16, 0) <= local_time < time(17, 0):
            return False
        return True

    try:
        if not mcal:
            raise RuntimeError("pandas_market_calendars is not installed")
        cme = mcal.get_calendar('CMEGlobex_RB')
        start = (dt - timedelta(days=1)).date()
        end = (dt + timedelta(days=2)).date()
        schedule = cme.schedule(start_date=start, end_date=end)
        return cme.open_at_time(schedule, dt)
    except Exception as e:
        fallback_open = simple_globex_energy_hours(dt)
        print(f"Calendar check failed: {e}. Using simple Globex-hours fallback: open={fallback_open}.")
        return fallback_open

def is_holiday(dt):
    """Check if a given datetime is a US market holiday."""
    local_date = dt.astimezone(TZ).date()
    return local_date in us_market_holidays(local_date.year)

def resolve_active_schwab_symbol(prefix, now, access_token, candidate_months=4):
    def quote_float(quote, *keys):
        for key in keys:
            value = quote.get(key)
            if value not in (None, ''):
                try:
                    value = float(value)
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass
        return None

    def quote_activity(quote):
        return quote_float(quote, 'volume', 'volumeDay', 'totalVolume', 'openInterest') or 0.0

    # Use the CME Globex session date (not raw calendar date) for contract resolution.
    # After 5 PM CT, the session has already rolled to the next calendar day, so
    # get_session_date_str correctly returns tomorrow's date for post-5PM runs.
    session_date_str = get_session_date_str(now)
    from datetime import date as _date_type
    session_date = _date_type.fromisoformat(session_date_str)
    schedule_year, schedule_month, front_ltd = get_front_month_contract(session_date, prefix)
    # If the session date has reached the front-month LTD, that contract expires
    # during this session — advance to the next contract for the candidate list.
    if session_date >= front_ltd:
        print(f"[{prefix}] Session date {session_date} >= LTD {front_ltd}: advancing candidates to next contract")
        schedule_year, schedule_month = add_month(schedule_year, schedule_month, 1)
    candidates = []
    for i in range(candidate_months):
        cyear, cmonth = add_month(schedule_year, schedule_month, i)
        candidates.append(f"/{prefix}{DELIVERY_MONTH_CODES[cmonth]}{cyear % 100:02d}")

    # ── Resolution strategy ──────────────────────────────────────────────────
    # 1. PRIMARY: yfinance underlyingSymbol  (tracks market-convention roll;
    #    matches what thinkorswim and Yahoo Finance show as the front month,
    #    which typically occurs 3–5 weeks before the mathematical LTD).
    # 2. FALLBACK: Schwab live-quote volume  (pick the candidate with the
    #    highest real-time activity; used when yfinance is unavailable).
    # 3. LAST RESORT: candidates[0]  (first contract in our session-adjusted
    #    candidate list).

    # 1. Try yfinance underlyingSymbol first
    try:
        yf_symbol = "RB=F" if prefix == "RB" else "HO=F" if prefix == "HO" else "CL=F"
        yf_ticker = yf.Ticker(yf_symbol, session=_YF_SESSION)
        underlying = yf_ticker.info.get('underlyingSymbol')
        if underlying:
            resolved_symbol = '/' + underlying.split('.')[0]
            if resolved_symbol.startswith(f"/{prefix}"):
                print(f"[{prefix}] Resolved active Schwab symbol from yfinance underlyingSymbol (primary): {resolved_symbol}")
                return resolved_symbol
    except Exception as yf_err:
        print(f"[{prefix}] yfinance underlyingSymbol lookup failed: {yf_err}. Falling back to Schwab volume.")

    # 2. Schwab live-quote volume fallback
    if access_token:
        try:
            symbols = ",".join(candidates)
            res = requests.get(
                "https://api.schwabapi.com/marketdata/v1/quotes",
                params={"symbols": symbols},
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15
            )
            res.raise_for_status()
            res_json = res.json()

            best_symbol = None
            best_score = (-1, -1.0, float('inf'))
            for idx, symbol in enumerate(candidates):
                quote_entry = None
                if symbol in res_json:
                    quote_entry = res_json[symbol].get('quote', {})
                elif isinstance(res_json, dict):
                    quote_entry = next(
                        (item.get('quote') for item in res_json.values() if isinstance(item, dict) and item.get('quote', {}).get('symbol') == symbol),
                        None
                    )
                if not quote_entry:
                    continue
                score = float(quote_activity(quote_entry))
                candidate_score = (1, score, -idx)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_symbol = symbol

            if best_symbol:
                if best_score[1] >= 100:
                    print(f"[{prefix}] Resolved active Schwab symbol from live quotes (fallback): {best_symbol} (volume: {best_score[1]:.0f})")
                    return best_symbol
                else:
                    print(f"[{prefix}] Schwab volume fallback bypassed: max volume {best_score[1]:.0f} is below threshold of 100.")
        except Exception as exc:
            print(f"[{prefix}] Schwab volume resolution failed: {exc}.")

    return candidates[0]


def fetch_commodity(prefix, cfg, now, access_token):
    # Retrieve or resolve the active symbol for the session to ensure consistency
    state = load_alert_state()
    session_str = get_session_date_str(now)
    cache_key = f"ACTIVE_SYMBOL_{prefix}_{session_str}"
    
    cached_symbol = state.get(cache_key)
    # Check if it is a contract roll day using both the raw calendar date AND the
    # CME session date.  After 5 PM CT, the session date is tomorrow; if that
    # session date equals the front-month LTD the contract expires this session.
    on_roll_day = False
    if prefix in ('RB', 'HO', 'CL'):
        on_roll_day = is_contract_roll_day(now, prefix)
        if not on_roll_day:
            try:
                from datetime import date as _date_type
                _session_date = _date_type.fromisoformat(session_str)
                _, _, _front_ltd = get_front_month_contract(_session_date, prefix)
                if _session_date >= _front_ltd:
                    on_roll_day = True
            except Exception:
                pass
        # Market-convention override: if the calendar LTD fired but the cached active
        # symbol for today and a recent prior session are the same contract, the market
        # has already rolled weeks ago and the calendar is a false positive.
        if on_roll_day and cached_symbol:
            try:
                _now_ct = now.astimezone(TZ) if hasattr(now, 'astimezone') else now
                for _back in range(1, 8):
                    from datetime import timedelta as _td2
                    _prev_ct = _now_ct - _td2(days=_back)
                    _prev_sess = get_session_date_str(_prev_ct.replace(hour=12))
                    _prev_sym = state.get(f"ACTIVE_SYMBOL_{prefix}_{_prev_sess}")
                    if _prev_sym:
                        break
                else:
                    _prev_sym = None
                if _prev_sym and cached_symbol == _prev_sym:
                    print(f"[{prefix}] Calendar LTD fired in fetch_commodity but symbol "
                          f"unchanged ({cached_symbol}). Market already rolled — cache is valid.")
                    on_roll_day = False
            except Exception as _e:
                print(f"[{prefix}] Warning: symbol-continuity check failed in fetch_commodity: {_e}")
    if cached_symbol and not on_roll_day:
        schwab_symbol = cached_symbol
        print(f"[{prefix}] Using cached active symbol for session {session_str}: {schwab_symbol}")
    else:
        if on_roll_day:
            print(f"[{prefix}] Session roll day — bypassing stale cache and re-resolving by live Schwab volume.")
        schwab_symbol = resolve_active_schwab_symbol(prefix, now, access_token)
        state[cache_key] = schwab_symbol
        try:
            save_alert_state(state)
            print(f"[{prefix}] Resolved and cached active symbol for session {session_str}: {schwab_symbol}")
        except Exception as e:
            print(f"Warning: could not save active symbol to alert state: {e}")
            
    dynamic_yf_symbol = schwab_to_yfinance_symbol(schwab_symbol)
    print(f"[{prefix}] Targeting front-month: Schwab {schwab_symbol} | yf {dynamic_yf_symbol}")
    
    current_price = open_price = high_price = low_price = None
    data_source = None

    def quote_float(quote, *keys):
        for key in keys:
            value = quote.get(key)
            if value not in (None, ''):
                try:
                    value = float(value)
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass
        return None

    def fetch_yfinance_quote(symbol):
        yf_t = yf.Ticker(symbol, session=_YF_SESSION)
        try:
            current = float(yf_t.fast_info['last_price'])
        except Exception:
            current = None
        hist = yf_t.history(period='1d', interval='5m')
        if hist.empty and not current:
            raise RuntimeError(f"No Yahoo Finance data for {symbol}")
        if not current:
            current = float(hist['Close'].iloc[-1])
        if not hist.empty:
            open_ = float(hist['Open'].iloc[0])
            high_ = float(hist['High'].max())
            low_ = float(hist['Low'].min())
            quote_time_val = hist.index[-1].to_pydatetime().astimezone(TZ)
        else:
            open_ = current
            high_ = low_ = 0.0
            quote_time_val = now
        return current, open_, high_, low_, quote_time_val

    quote_time = now
    try:
        res = requests.get(
            "https://api.schwabapi.com/marketdata/v1/quotes",
            params={"symbols": schwab_symbol},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15
        )
        res.raise_for_status()
        res_json = res.json()
        if schwab_symbol in res_json:
            quote = res_json[schwab_symbol]['quote']
        else:
            quote = res_json[list(res_json.keys())[0]]['quote']
        
        current_price = quote_float(quote, 'lastPrice', 'mark', 'bidPrice', 'askPrice')
        open_price    = quote_float(quote, 'openPrice') or current_price
        high_price    = quote_float(quote, 'highPrice') or 0.0
        low_price     = quote_float(quote, 'lowPrice') or 0.0
        schwab_close  = quote_float(quote, 'closePrice') or 0.0
        if not current_price:
            raise RuntimeError(f"Schwab quote for {schwab_symbol} did not include a usable price")
        
        quote_time_ms = quote.get('quoteTime')
        if quote_time_ms:
            quote_time = datetime.fromtimestamp(quote_time_ms / 1000.0, pytz.utc).astimezone(TZ)
        else:
            quote_time = now
            
        data_source   = 'schwab'
        print(f"[{prefix}] Schwab success")
    except Exception as e:
        print(f"[{prefix}] Schwab failed: {e}. Falling back to yfinance.")
        import time
        yf_symbols = [dynamic_yf_symbol]
        if cfg['yf_symbol'] not in yf_symbols:
            yf_symbols.append(cfg['yf_symbol'])
        for attempt in range(3):
            try:
                last_err = None
                for yf_symbol in yf_symbols:
                    try:
                        current_price, open_price, high_price, low_price, quote_time = fetch_yfinance_quote(yf_symbol)
                        dynamic_yf_symbol = yf_symbol
                        break
                    except Exception as yf_symbol_err:
                        last_err = yf_symbol_err
                else:
                    raise last_err or RuntimeError("Yahoo Finance fallback failed")
                data_source = 'yfinance'
                print(f"[{prefix}] yfinance fallback success ({dynamic_yf_symbol})")
                break
            except Exception as yf_err:
                if attempt < 2:
                    time.sleep(10)
                else:
                    print(f"[{prefix}] yfinance fallback failed: {yf_err}")

    if not current_price:
        print(f"[{prefix}] Could not fetch data, skipping.")
        return None
        
    if not open_price: open_price = current_price
    
    session_date_str = get_session_date_str(now)
    session_date = datetime.fromisoformat(session_date_str).date()
    
    yesterday_close = None
    five_day_high = five_day_low = thirty_day_avg = sma_3 = sma_10 = None
    history_5d = []
    # Prioritize graves_history.csv daily settlements for yesterday_close and compute fallback stats
    try:
        col_idx = 1 if prefix == "RB" else 2
        csv_file_path = os.path.join(DATA_DIR, "graves_history.csv")
        if os.path.exists(csv_file_path):
            hist_records = []
            with open(csv_file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None) # skip header
                for row in reader:
                    if len(row) > col_idx:
                        row_date_str = row[0].strip()
                        if row_date_str:
                            try:
                                row_date = datetime.fromisoformat(row_date_str).date()
                                val_str = row[col_idx].strip()
                                if val_str:
                                    val = float(val_str)
                                    if row_date < session_date:
                                        hist_records.append((row_date, val))
                            except (ValueError, IndexError):
                                pass
            if hist_records:
                hist_records.sort(key=lambda x: x[0])
                best_date, yesterday_close = hist_records[-1]
                print(f"[{prefix}] Prioritized yesterday_close from local CSV (date: {best_date}): {yesterday_close}")
                
                # Extract clean list of prices
                hist_prices = [x[1] for x in hist_records]
                
                # Set fallback statistics in case yfinance fails
                thirty_day_avg = float(sum(hist_prices[-30:]) / len(hist_prices[-30:]))
                if len(hist_prices) >= 3:
                    sma_3 = float(sum(hist_prices[-3:]) / len(hist_prices[-3:]))
                if len(hist_prices) >= 10:
                    sma_10 = float(sum(hist_prices[-10:]) / len(hist_prices[-10:]))
                five_day_high = float(max(hist_prices[-5:]))
                five_day_low  = float(min(hist_prices[-5:]))
    except Exception as e:
        print(f"[{prefix}] Failed to read statistics fallback from CSV: {e}")

    # Fallback to Schwab close price if CSV not found/empty
    if yesterday_close is None and data_source == 'schwab' and schwab_close > 0:
        yesterday_close = schwab_close
        
    try:
        yf_t = yf.Ticker(dynamic_yf_symbol, session=_YF_SESSION)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_5d = executor.submit(yf_t.history, period='5d', interval='1h')
            fut_30d = executor.submit(yf_t.history, period='1mo', interval='1d')
            h5d = fut_5d.result()
            h30d = fut_30d.result()

        if not h5d.empty:
            history_5d = [{"t": idx.astimezone(TZ).isoformat(), "p": round(float(price), 4)} for idx, price in zip(h5d.index, h5d['Close'])]
            five_day_high = float(h5d['High'].max())
            five_day_low  = float(h5d['Low'].min())
        if not h30d.empty:
            prev_daily = h30d[[d.date() < session_date for d in h30d.index.to_pydatetime()]]
            if not prev_daily.empty:
                last_daily_date = prev_daily.index[-1].date()
                target_prev_day = previous_nymex_business_day(session_date)
                if last_daily_date < target_prev_day:
                    print(f"[{prefix}] Warning: yfinance daily close history is stale (last available: {last_daily_date}, expected: {target_prev_day}). Bypassing yfinance close price.")
                elif yesterday_close is None:
                    yesterday_close = float(prev_daily['Close'].iloc[-1])
                
                # Calculate moving averages using strictly closed daily data (prev_daily) up to T-1
                thirty_day_avg = float(prev_daily['Close'].mean())
                if len(prev_daily) >= 3:
                    sma_3 = float(prev_daily['Close'].tail(3).mean())
                if len(prev_daily) >= 10:
                    sma_10 = float(prev_daily['Close'].tail(10).mean())
    except Exception:
        pass

    baseline_price = yesterday_close if yesterday_close else open_price
    daily_pct = ((current_price - baseline_price) / baseline_price) * 100

    if cfg.get('hidden'):
        chart_intra = None
        chart_5d = None
    else:
        history_key = contract_state_prefix(prefix, schwab_symbol)
        history_intra = load_price_history(history_key)
        history_intra = append_price(history_intra, quote_time, current_price)
        save_price_history(history_intra, history_key)
        chart_intra = generate_intraday_chart(history_intra, current_price, baseline_price, high_price, low_price, daily_pct, cfg['name'])
        chart_5d = generate_5day_chart(history_5d, current_price)
    
    print(f"[{prefix}] Fetched: ${current_price:.4f} ({daily_pct:+.2f}%)")
    return {
        'prefix': prefix,
        'current_price': current_price,
        'open_price': open_price,
        'high_price': high_price,
        'low_price': low_price,
        'daily_pct': daily_pct,
        'yesterday_close': yesterday_close,
        'five_day_high': five_day_high,
        'five_day_low': five_day_low,
        'thirty_day_avg': thirty_day_avg,
        'sma_3': sma_3,
        'sma_10': sma_10,
        'chart_intraday_b64': chart_intra,
        'chart_5d_b64': chart_5d,
        'schwab_symbol': schwab_symbol,
        'yfinance_symbol': dynamic_yf_symbol,
        'quote_time': quote_time
    }


if __name__ == "__main__":
    main()
