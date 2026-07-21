import os
import sys
import json
import csv
import math
import imaplib
import email
import email.utils
import re
import smtplib
import subprocess
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
import validate_data


GMAIL_USER = os.environ.get('GMAIL_USER')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
TO_EMAIL = os.environ.get('TO_EMAIL')
GRAVES_EMAIL = os.environ.get('GRAVES_EMAIL')
GRAVES_APP_PASSWORD = os.environ.get('GRAVES_APP_PASSWORD')
TZ = pytz.timezone('America/Chicago')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "graves_history.csv")
DS_PATH = os.path.join(DATA_DIR, "daily_settlement.json")
SETTLEMENT_PROVENANCE_COLUMNS = [
    "session_date", "commodity", "settlement_price", "schwab_symbol",
    "yfinance_symbol", "source", "captured_at", "provenance_status",
]


def settlement_provenance_path():
    return os.path.join(os.path.dirname(CSV_PATH), "nymex_settlement_provenance.csv")

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

def send_alert_email(subject, body):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print("Cannot send alert email: missing credentials.")
        return
        
    email_to_env = os.environ.get('TO_EMAIL', '')
    recipients = [e.strip() for e in email_to_env.split(',') if e.strip()]

    # Deduplicate keeping order
    seen = set()
    emails = []
    for r in recipients:
        if r not in seen:
            seen.add(r)
            emails.append(r)
            
    if not emails:
        emails = [GMAIL_USER]
        
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = GMAIL_USER
    msg['To'] = ", ".join(emails)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, emails, msg.as_string())
        print(f"Alert email sent to {mask_recipient(emails)}: {subject}")
    except Exception as e:
        print(f"Failed to send alert email: {mask_sensitive_text(e)}")

def git_pull_rebase():
    try:
        subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        # Stash any temporary changes (like integrity_hashes updating locally) to ensure a clean pull
        has_changes = subprocess.run(["git", "diff", "--quiet"], capture_output=True).returncode != 0
        if has_changes:
            subprocess.run(["git", "stash"], check=True)
        subprocess.run(["git", "pull", "--rebase"], check=True)
        if has_changes:
            subprocess.run(["git", "stash", "pop"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git pull failed: {e}")
        sys.exit(1)

def git_commit_push(message):
    subprocess.run([
        "git", "add", "data/graves_history.csv", "data/nymex_settlement_provenance.csv",
        "data/integrity_hashes.csv"
    ], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)
    print("Git commit and push successful.")

LABELS = {
    "rack_u": "E10 - UNLEADED",
    "rack_p": "E10 - PREMIUM",
    "rack_d": "CLEAR DIESEL",
}

def extract_price_near_label(text, label):
    """Find label in text; return the first price number within 60 chars AFTER it on the same line only."""
    # Collapses all forms of whitespace (including \xa0 and tabs) to standard spaces
    normalized_label = re.sub(r'\s+', ' ', label).lower()
    price_pattern = re.compile(r'\$?(\d+\.\d{2,5})')
    
    for line in text.splitlines():
        normalized_line = re.sub(r'\s+', ' ', line)
        pos = normalized_line.lower().find(normalized_label)
        if pos != -1:
            # Search only within 60 characters AFTER the label on this line
            after_label = normalized_line[pos + len(normalized_label): pos + len(normalized_label) + 60]
            m = price_pattern.search(after_label)
            if m:
                return float(m.group(1))
    return None

def check_inbox_for_prices(target_date_str):
    graves_email = os.environ.get('GRAVES_EMAIL')
    graves_pass = os.environ.get('GRAVES_APP_PASSWORD')
    if not graves_email or not graves_pass:
        print("Missing GRAVES_EMAIL or GRAVES_APP_PASSWORD. Cannot check invoices.")
        return None, None

    price_min = 1.50
    price_max = 6.00
    try:
        config_path = os.path.join(DATA_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                price_min = cfg.get("PRICE_MIN", 1.50)
                price_max = cfg.get("PRICE_MAX", 6.00)
    except Exception as cfg_e:
        print(f"Error loading price bounds config: {mask_sensitive_text(cfg_e)}")
        
    import time
    mail = None
    for attempt in range(3):
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=15)
            mail.login(graves_email, graves_pass)
            break
        except Exception as conn_e:
            print(f"IMAP connection/login attempt {attempt + 1} failed: {mask_sensitive_text(conn_e)}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return None, None

    try:
        mail.select('"[Gmail]/All Mail"')
        
        # Look back 2 days to handle any timezone/date boundary differences
        since_date = (datetime.now(TZ) - timedelta(days=2)).strftime("%d-%b-%Y")
        
        # Broadened search: match subject keywords first, then fall back to domain
        search_query = f'(SINCE "{since_date}" SUBJECT "Latest prices")'
        status, messages = mail.search(None, search_query)

        if status != "OK" or not messages[0]:
            # Try a looser domain-based search as a fallback
            alt_query = f'(SINCE "{since_date}" FROM "gravesoil.com")'
            status2, messages2 = mail.search(None, alt_query)
            if status2 == "OK" and messages2[0]:
                messages = messages2
                print(f"IMAP search fell back to domain search. Query: {mask_sensitive_text(alt_query)}")
            else:
                print(f"IMAP search returned no messages. Status: {status}, Query: {mask_sensitive_text(search_query)}")
                return None, None

        # Process the most recent email first
        for num in reversed(messages[0].split()):
            try:
                status, data = mail.fetch(num, '(RFC822)')
                msg = email.message_from_bytes(data[0][1])

                # Helpful debug info: show From/Subject/Date to action logs
                from_hdr = msg.get('From')
                subject = msg.get('Subject')
                email_date = email.utils.parsedate_to_datetime(msg.get('Date'))
                print(f"Examining email From: {mask_sensitive_text(from_hdr)} Subject: {mask_sensitive_text(subject)} Date: {email_date.astimezone(TZ) if email_date else 'None'}")

                # Verify that the email belongs to the target date
                if not email_date:
                    continue
                local_date_str = email_date.astimezone(TZ).date().isoformat()

                # Allow email date to be either:
                # 1. Exactly target_date_str (normal case)
                # 2. The day after target_date_str, before 9:00 AM CT (late case)
                is_exact_match = (local_date_str == target_date_str)
                is_late_morning_match = False
                try:
                    from datetime import date as _date
                    target_date = _date.fromisoformat(target_date_str)
                    email_local_date = email_date.astimezone(TZ).date()
                    delta_days = (email_local_date - target_date).days
                    email_hour = email_date.astimezone(TZ).hour
                    is_late_morning_match = (delta_days == 1 and email_hour < 9)
                except Exception:
                    pass

                if not (is_exact_match or is_late_morning_match):
                    continue

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get('Content-Disposition'))
                        if ctype == 'text/plain' and 'attachment' not in cdispo:
                            body += part.get_payload(decode=True).decode('utf-8', errors='ignore') + "\n"
                        elif ctype == 'text/html' and 'attachment' not in cdispo:
                            html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            body += BeautifulSoup(html, "html.parser").get_text(separator=" ") + "\n"
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body_decoded = payload.decode('utf-8', errors='ignore')
                        if msg.get_content_type() == 'text/html':
                            body = BeautifulSoup(body_decoded, "html.parser").get_text(separator=" ")
                        else:
                            body = body_decoded

                # Verify target date matches the 'good from' date in the email body if present
                body_date_match = re.search(r'good\s+from\s+(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', body, re.IGNORECASE)
                if body_date_match:
                    b_month = int(body_date_match.group(1))
                    b_day = int(body_date_match.group(2))
                    b_year = int(body_date_match.group(3))
                    if b_year < 100:
                        b_year += 2000
                    body_date_str = f"{b_year:04d}-{b_month:02d}-{b_day:02d}"
                    print(f"Parsed 'good from' date from email body: {body_date_str}")
                    if body_date_str != target_date_str:
                        print(f"Email body date {body_date_str} does not match target date {target_date_str}. Skipping this email.")
                        continue
                else:
                    print("Could not find 'good from' date in email body. Falling back to email header date matching.")

                # Primary extraction: labels near known markers
                prices = {}
                for key, label in LABELS.items():
                    p = extract_price_near_label(body, label)
                    if p is None or not (price_min <= p <= price_max):
                        print(f"Label extraction failed for '{label}' (got {p}). Skipping this email.")
                        prices = {}
                        break
                    prices[key] = p

                if len(prices) == 3:
                    rack_u = prices['rack_u']
                    rack_p = prices['rack_p']
                    rack_d = prices['rack_d']
                    print(f"Parsed prices from email: Unleaded={rack_u}, Premium={rack_p}, Diesel={rack_d}")
                    return target_date_str, (rack_u, rack_p, rack_d)

            except Exception as loop_e:
                print(f"Skipping badly formatted email {num}: {mask_sensitive_text(loop_e)}")
                continue

        return None, None
    except Exception as e:
        print(f"IMAP Error: {mask_sensitive_text(e)}")
        return None, None

def read_daily_settlement(target_date_str):
    if not os.path.exists(DS_PATH):
        return None
    with open(DS_PATH, "r") as f:
        ds = json.load(f)
    if ds.get("date") == target_date_str:
        return ds
    return None


def append_settlement_provenance(ds, target_date_str):
    """Persist the contract identity for each settlement archived in graves_history."""
    source = ds.get("source", "daily_settlement")
    captured_at = ds.get("captured_at", "")
    rows = []
    for commodity, price_key, contract_key, yf_key, source_key in (
        ("RB", "rbob_settlement", "rbob_contract", "rbob_yfinance_symbol", "rbob_source"),
        ("HO", "heating_oil_settlement", "heating_oil_contract", "heating_oil_yfinance_symbol", "heating_oil_source"),
    ):
        price = ds.get(price_key, "")
        if price in (None, ""):
            continue
        contract = ds.get(contract_key, "") or ""
        rows.append({
            "session_date": target_date_str,
            "commodity": commodity,
            "settlement_price": f"{float(price):.4f}",
            "schwab_symbol": contract,
            "yfinance_symbol": ds.get(yf_key, "") or "",
            "source": ds.get(source_key, source),
            "captured_at": captured_at,
            "provenance_status": "verified" if contract else "unknown",
        })

    if not rows:
        return
    provenance_path = settlement_provenance_path()
    existing = set()
    if os.path.exists(provenance_path):
        with open(provenance_path, newline="") as f:
            existing = {
                (row.get("session_date"), row.get("commodity"))
                for row in csv.DictReader(f)
            }
    write_header = not os.path.exists(provenance_path)
    with open(provenance_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SETTLEMENT_PROVENANCE_COLUMNS, lineterminator="\n"
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            if (row["session_date"], row["commodity"]) not in existing:
                writer.writerow(row)

def get_github_settlement_snapshots(target_date_str):
    gh_pat = os.environ.get('GH_PAT')
    gh_repo = os.environ.get('GH_REPO')
    # First, allow direct env-var overrides (useful for CI secrets or manual runs)
    # Support either plain numeric values or JSON strings like '{"date":"2026-05-28","price":3.0659}'
    try:
        rb_env = os.environ.get('SETTLE_SNAPSHOT_RB')
        ho_env = os.environ.get('SETTLE_SNAPSHOT_HO')
        if rb_env and ho_env:
            def parse_snapshot(env_val):
                env_val = env_val.strip()
                # Try parse as JSON
                try:
                    parsed = json.loads(env_val)
                    if isinstance(parsed, dict):
                        if parsed.get('date') == target_date_str and 'price' in parsed:
                            return float(parsed['price'])
                        # support legacy key names
                        if parsed.get('date') == target_date_str and 'rbob_settlement' in parsed:
                            return float(parsed['rbob_settlement'])
                    # If JSON is just a number or string, fall through
                except Exception:
                    pass
                # Try parse as plain float
                try:
                    return float(env_val)
                except Exception:
                    return None

            rb_val = parse_snapshot(rb_env)
            ho_val = parse_snapshot(ho_env)
            if rb_val is not None and ho_val is not None:
                return {
                    "date": target_date_str,
                    "rbob_settlement": rb_val,
                    "heating_oil_settlement": ho_val,
                    "source": "env_override"
                }
    except Exception as e:
        print(f"Error parsing SETTLE_SNAPSHOT_* env vars: {e}")
    if not gh_pat or not gh_repo:
        print("GitHub credentials missing from environment. Skipping GitHub variable lookup.")
        return None
        
    import requests
    headers = {
        "Authorization": f"token {gh_pat}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        # Fetch SETTLE_SNAPSHOT_RB and SETTLE_SNAPSHOT_HO
        rb_url = f"https://api.github.com/repos/{gh_repo}/actions/variables/SETTLE_SNAPSHOT_RB"
        ho_url = f"https://api.github.com/repos/{gh_repo}/actions/variables/SETTLE_SNAPSHOT_HO"
        
        rb_res = requests.get(rb_url, headers=headers, timeout=10)
        ho_res = requests.get(ho_url, headers=headers, timeout=10)
        
        rb_val, ho_val = None, None
        rb_snap, ho_snap = {}, {}

        if rb_res.status_code == 200:
            rb_snap = json.loads(rb_res.json().get("value", "{}"))
            if rb_snap.get("date") == target_date_str:
                rb_val = float(rb_snap["price"])
                if rb_snap.get("schwab_symbol"):
                    print(f"[RB] GitHub variables success: {rb_snap['schwab_symbol']} price={rb_val}")

        if ho_res.status_code == 200:
            ho_snap = json.loads(ho_res.json().get("value", "{}"))
            if ho_snap.get("date") == target_date_str:
                ho_val = float(ho_snap["price"])
                if ho_snap.get("schwab_symbol"):
                    print(f"[HO] GitHub variables success: {ho_snap['schwab_symbol']} price={ho_val}")

        if rb_val is not None and ho_val is not None:
            return {
                "date": target_date_str,
                "rbob_settlement": rb_val,
                "heating_oil_settlement": ho_val,
                "source": "github_variables",
                "captured_at": rb_snap.get("captured_at") or ho_snap.get("captured_at", ""),
                "rbob_contract": rb_snap.get("schwab_symbol", ""),
                "heating_oil_contract": ho_snap.get("schwab_symbol", ""),
                "rbob_yfinance_symbol": rb_snap.get("yfinance_symbol", ""),
                "heating_oil_yfinance_symbol": ho_snap.get("yfinance_symbol", ""),
                "rbob_source": rb_snap.get("source", "github_variables"),
                "heating_oil_source": ho_snap.get("source", "github_variables"),
            }
        else:
            print("GitHub variables for settlement snapshots were missing, stale, or incomplete.")
    except Exception as e:
        print(f"Failed to fetch settlement snapshots from GitHub variables: {e}")
        
    return None



def get_thinkorswim_settlement(target_date_str):
    """Attempt to obtain settlement snapshots from ThinkOrSwim exports.
    Supports three mechanisms (in order):
    - `TOS_CSV_PATH` env var pointing to a local CSV with columns date,rbob,ho
    - `TOS_SETTLEMENT_URL` env var: an HTTP GET that returns JSON with keys
      `date`, `rbob_settlement`, `heating_oil_settlement`. Optionally use
      `TOS_API_KEY` as a Bearer token header.
    - If not configured or fails, returns None.
    """
    # 0) Prefer Schwab/ThinkOrSwim API if credentials are available
    SCHWAB_APP_KEY = os.environ.get('SCHWAB_APP_KEY')
    SCHWAB_APP_SECRET = os.environ.get('SCHWAB_APP_SECRET')
    SCHWAB_REFRESH_TOKEN = os.environ.get('SCHWAB_REFRESH_TOKEN')
    if SCHWAB_APP_KEY and SCHWAB_APP_SECRET and SCHWAB_REFRESH_TOKEN:
        try:
            import base64
            import requests
            from datetime import datetime as _dt
            auth_header = base64.b64encode(f"{SCHWAB_APP_KEY}:{SCHWAB_APP_SECRET}".encode()).decode()
            token_res = requests.post(
                "https://api.schwabapi.com/v1/oauth/token",
                data={"grant_type": "refresh_token", "refresh_token": SCHWAB_REFRESH_TOKEN},
                headers={"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            token_res.raise_for_status()
            token_json = token_res.json()
            access_token = token_json.get('access_token')
            if access_token:
                # Resolve the front-month Schwab symbols using market-convention first
                # (yfinance underlyingSymbol), then fall back to math-based LTD schedule.
                # The math-based approach lags the real market roll by 3-5 weeks, causing
                # wrong-contract settlement prices to be stored during the rollover window.
                try:
                    dt_target = _dt.fromisoformat(target_date_str)
                except Exception:
                    dt_target = _dt.now()

                def _resolve_ingest_symbol(prefix):
                    """Resolve front-month contract symbol: yfinance first, math fallback."""
                    from futures_util import get_front_month_schwab_symbol
                    import yfinance as _yf
                    yf_root = "RB=F" if prefix == "RB" else "HO=F"
                    try:
                        underlying = _yf.Ticker(yf_root).info.get('underlyingSymbol')
                        if underlying and underlying.startswith(prefix):
                            resolved = '/' + underlying.split('.')[0]
                            print(f"[ingest] {prefix} front-month resolved via yfinance underlyingSymbol: {resolved}")
                            return resolved
                    except Exception as _yf_err:
                        print(f"[ingest] {prefix} yfinance underlyingSymbol lookup failed: {_yf_err}. Using math fallback.")
                    math_sym = get_front_month_schwab_symbol(dt_target, prefix)
                    print(f"[ingest] {prefix} front-month resolved via math fallback: {math_sym}")
                    return math_sym

                rb_sym = _resolve_ingest_symbol('RB')
                ho_sym = _resolve_ingest_symbol('HO')

                # Use pricehistory endpoint to get historical daily candles
                rb_val = None
                ho_val = None

                for symbol in (rb_sym, ho_sym):
                    try:
                        ph_res = requests.get(
                            "https://api.schwabapi.com/marketdata/v1/pricehistory",
                            params={
                                "symbol": symbol,
                                "periodType": "month",
                                "period": 1,
                                "frequencyType": "daily",
                                "frequency": 1
                            },
                            headers={"Authorization": f"Bearer {access_token}"},
                            timeout=10
                        )
                        ph_res.raise_for_status()
                        ph_data = ph_res.json()

                        # Look for candle matching target date
                        candles = ph_data.get('candles', [])
                        for candle in candles:
                            # datetime is in milliseconds (epoch)
                            candle_dt = _dt.fromtimestamp(candle.get('datetime', 0) / 1000)
                            candle_date_str = candle_dt.date().isoformat()
                            if candle_date_str == target_date_str:
                                close_val = candle.get('close')
                                if close_val is not None:
                                    close_val = float(close_val)
                                    if symbol == rb_sym:
                                        rb_val = close_val
                                    elif symbol == ho_sym:
                                        ho_val = close_val
                                if os.environ.get('INGEST_DEBUG'):
                                    print(f"Schwab pricehistory candle for {symbol} on {target_date_str}: close={close_val}")
                                break
                        else:
                            if os.environ.get('INGEST_DEBUG'):
                                print(f"No candle found in pricehistory for {symbol} on {target_date_str}")
                    except Exception as e:
                        if os.environ.get('INGEST_DEBUG'):
                            print(f"Schwab pricehistory fetch for {symbol} failed: {e}")
                        continue

                if rb_val is not None and ho_val is not None:
                    return {
                        'date': target_date_str,
                        'rbob_settlement': round(rb_val, 4),
                        'heating_oil_settlement': round(ho_val, 4),
                        'source': 'schwab_pricehistory'
                    }
        except Exception as e:
            print(f'ThinkOrSwim/Schwab API fetch failed: {e}')

    # 1) Local CSV path (useful for CI tests or manual overrides)
    tos_csv = os.environ.get('TOS_CSV_PATH')
    if tos_csv:
        try:
            if os.path.exists(tos_csv):
                with open(tos_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('date') == target_date_str:
                            try:
                                # Support several common key names that may represent
                                # the settlement/close price coming from ThinkOrSwim exports.
                                def _get_numeric(keys):
                                    for kk in keys:
                                        if kk in row and row.get(kk) not in (None, ''):
                                            try:
                                                return float(row.get(kk))
                                            except Exception:
                                                continue
                                    return None
                                rb = _get_numeric(['rbob', 'rbob_settlement', 'rbob_close', 'rbob_settle', 'rbob_price'])
                                ho = _get_numeric(['ho', 'heating_oil_settlement', 'heating_oil_close', 'ho_close', 'ho_settle', 'heating_oil_price'])
                                return {
                                    'date': target_date_str,
                                    'rbob_settlement': round(rb, 4) if rb is not None else None,
                                    'heating_oil_settlement': round(ho, 4) if ho is not None else None,
                                    'source': 'tos_csv'
                                }
                            except Exception:
                                continue
        except Exception as e:
            print(f'TOS CSV parse error: {e}')

    # 2) Remote HTTP endpoint
    tos_url = os.environ.get('TOS_SETTLEMENT_URL')
    if tos_url:
        try:
            import requests
            headers = {}
            tos_key = os.environ.get('TOS_API_KEY')
            if tos_key:
                headers['Authorization'] = f'Bearer {tos_key}'

            resp = requests.get(tos_url, params={'date': target_date_str}, headers=headers, timeout=10)
            if resp.status_code == 200:
                try:
                    j = resp.json()
                    # Accept either top-level keys or nested snapshot
                    if isinstance(j, dict):
                        # normalize keys
                        def _pick_from_json(dct, keys):
                            for kk in keys:
                                v = dct.get(kk)
                                if v not in (None, ''):
                                    try:
                                        return float(v)
                                    except Exception:
                                        continue
                            return None

                        rb = _pick_from_json(j, ['rbob_settlement', 'rbob', 'price_rb', 'rbob_close', 'rbob_settle'])
                        ho = _pick_from_json(j, ['heating_oil_settlement', 'heating_oil', 'price_ho', 'ho_close', 'ho_settle'])
                        date = j.get('date') or j.get('snapshot_date') or target_date_str
                        if date == target_date_str and rb is not None and ho is not None:
                            return {
                                'date': date,
                                'rbob_settlement': round(rb, 4),
                                'heating_oil_settlement': round(ho, 4),
                                'source': 'tos_http'
                            }
                except Exception as je:
                    print(f'TOS HTTP JSON parse error: {je}')
            else:
                print(f'TOS HTTP fetch failed: status {resp.status_code} url={tos_url}')
        except Exception as e:
            print(f'Failed to fetch ThinkOrSwim settlement: {e}')

    return None


def ensure_trailing_newline(file_path):
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return
            f.seek(size - 1)
            last_byte = f.read(1)
        if last_byte != b'\n':
            print(f"File {file_path} is missing a trailing newline. Appending one...")
            with open(file_path, "ab") as f:
                f.write(b'\n')
    except Exception as e:
        print(f"Warning: Failed to ensure trailing newline for {file_path}: {e}")


def main():
    print("Starting price ingest...")
    git_pull_rebase()
    validate_data.validate_all(DATA_DIR)
    
    # Calculate target date based on Chicago timezone local hour
    now_local = datetime.now(TZ)
    if now_local.hour < 12:
        target_date_str = (now_local - timedelta(days=1)).date().isoformat()
    else:
        target_date_str = now_local.date().isoformat()
        
    print(f"Target date determined: {target_date_str} (local hour: {now_local.hour})")
    
    # Check if target date is a weekend (5 = Saturday, 6 = Sunday)
    target_dt = datetime.fromisoformat(target_date_str)
    if target_dt.weekday() in (5, 6):
        print(f"Target date {target_date_str} is weekend. Graves Oil is closed. Exiting silently.")
        sys.exit(0)
        
    # Read CSV to check for existing date (idempotency check)
    existing_dates = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if parts and parts[0]:
                    existing_dates.add(parts[0])
                    
    if target_date_str in existing_dates:
        print(f"Prices for {target_date_str} already ingested. Exiting silently.")
        sys.exit(0)
        
    # Query inbox for target date's email
    date_str, prices = check_inbox_for_prices(target_date_str)
    
    if not prices:
        # Determine retry vs. warning behavior
        current_hour = now_local.hour
        is_final_check = (current_hour == 0) or (current_hour < 4) or (current_hour in (8, 9))
        
        print(f"No valid price email found for {target_date_str}.")
        if is_final_check:
            time_label = "midnight" if current_hour < 4 else f"{current_hour}:00 AM"
            send_alert_email(
                "WARNING: Graves Oil Prices Missing",
                f"No Graves Oil prices received for {target_date_str} by {time_label}. Please check Graves Oil email/website manually."
            )
        else:
            print(f"Prices missing at hour {current_hour}. Will retry on next scheduled run.")
        sys.exit(0)

    print(f"Found prices for {date_str}: Unleaded={prices[0]}, Premium={prices[1]}, Diesel={prices[2]}")
    
    # Read CSV to check for duplicates (double check)
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(date_str + ","):
                    print("Date already exists in CSV. Skipping.")
                    sys.exit(0)
    
    # Get settlement data
    ds = read_daily_settlement(date_str)
    if not ds:
        print(f"daily_settlement.json missing or stale for {date_str}. Trying GitHub repository variables...")
        ds = get_github_settlement_snapshots(date_str)
        if ds:
            print(f"GitHub variables success: RB={ds['rbob_settlement']}, HO={ds['heating_oil_settlement']}")
            
    if not ds:
        print(f"GitHub variables missing or stale. Attempting ThinkOrSwim/Schwab fallback (no yfinance).")
        # Try ThinkOrSwim/Schwab export/source only (preferred reliable source)
        tos_ds = get_thinkorswim_settlement(date_str)
        if tos_ds:
            ds = tos_ds
            print(f"ThinkOrSwim fallback success: RB={ds['rbob_settlement']}, HO={ds['heating_oil_settlement']}")

    if not ds:
        # Settlement data is unavailable but we still have the Graves prices.
        # Write the row with empty settlement columns and send an informational
        # alert rather than aborting — losing the Graves prices is worse than
        # missing settlement for one day.
        print(f"Warning: could not obtain settlement data for {date_str}. Will write row with empty settlement.")
        send_alert_email(
            "Fuel Tracker Warning: Missing Settlement",
            f"Graves Oil prices were received for {date_str} and saved, but the "
            f"1:30 PM NYMEX settlement snapshot was unavailable from all sources "
            f"(daily_settlement.json, GitHub variables, and Yahoo Finance). "
            f"Settlement columns will be blank for this row. "
            f"You may need to fill them in manually."
        )
        ds = {"date": date_str, "rbob_settlement": "", "heating_oil_settlement": ""}

        
    rb_stl = ds.get("rbob_settlement", "")
    ho_stl = ds.get("heating_oil_settlement", "")

    # Write to CSV using retry transaction loop
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Ingestion transaction attempt {attempt + 1}/{max_retries}...")
        git_pull_rebase()
        
        # Double check if date was already written by another concurrent run that succeeded
        already_ingested = False
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r") as f:
                for line in f:
                    if line.startswith(date_str + ","):
                        already_ingested = True
                        break
        if already_ingested:
            print(f"Prices for {date_str} were already ingested by a concurrent run. Exiting silently.")
            sys.exit(0)
            
        # Ensure trailing newline exists in CSV before appending
        ensure_trailing_newline(CSV_PATH)
        # Write to CSV
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            
            def fmt_val(v):
                if v is None or v == "":
                    return ""
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        return ""
                    return f"{fv:.4f}"
                except (ValueError, TypeError):
                    return str(v)
            
            writer.writerow([
                date_str,
                fmt_val(rb_stl),
                fmt_val(ho_stl),
                f"{prices[0]:.4f}",
                f"{prices[1]:.4f}",
                f"{prices[2]:.4f}"
            ])

        append_settlement_provenance(ds, date_str)
            
        try:
            validate_data.validate_all(DATA_DIR)
        except Exception as val_e:
            print(f"Validation failed after write: {mask_sensitive_text(val_e)}")
            subprocess.run(["git", "reset", "--hard", "origin/main"])
            sys.exit(1)
            
        try:
            git_commit_push(f"Ingest Graves Oil prices for {date_str}")
            print("Successfully ingested.")
            break
        except Exception as push_e:
            print(f"Push conflict or commit failed on attempt {attempt + 1}: {mask_sensitive_text(push_e)}")
            subprocess.run(["git", "reset", "--hard", "origin/main"])
            if attempt == max_retries - 1:
                print("All retry attempts failed. Exiting.")
                sys.exit(1)
            import time
            import random
            time.sleep(2 ** attempt + random.uniform(0, 1))
    
    print("Triggering nightly backtester auto-tune...")
    try:
        result = subprocess.run(["python", "backtest.py"], capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Backtester crashed: {mask_sensitive_text(e)}")
        if e.stdout:
            print(f"Stdout:\n{mask_sensitive_text(e.stdout)}")
        if e.stderr:
            print(f"Stderr:\n{mask_sensitive_text(e.stderr)}")
        send_alert_email(
            "CRITICAL: Nightly calibration commit failed",
            f"The nightly backtester calibration failed to run or commit changes.\n\nError: {mask_sensitive_text(e)}\n\nStderr:\n{mask_sensitive_text(e.stderr)}\n\nPlease check GitHub Actions logs."
        )

# Trigger CI run to verify data consistency on main branch
if __name__ == "__main__":
    main()
