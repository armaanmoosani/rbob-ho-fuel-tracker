import imaplib
import email
import os
import re
import getpass
import pandas as pd
import pytz
from datetime import datetime
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup

# Graves posts its rack in the evening US Central time.  Every date derived
# from an email header must be expressed in this zone before it is used as a
# session date.
TZ = pytz.timezone("America/Chicago")

SENDER_SEARCH = "donotreply@gravesoil.com"
OUTPUT_CSV = "../data/graves_history.csv"

LABELS = {
    "rack_u": "E10 - UNLEADED",
    "rack_p": "E10 - PREMIUM",
    "rack_d": "CLEAR DIESEL",
}

def connect_imap(email_addr, app_pw):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_addr, app_pw)
    mail.select('"[Gmail]/All Mail"')
    return mail

def fetch_all_graves_emails(mail):
    search_query = f'FROM "{SENDER_SEARCH}"'
    status, messages = mail.search(None, search_query)
    if status != "OK":
        return []
    return messages[0].split()

def parse_email_date(msg):
    """Return the session date of a Graves email, in America/Chicago.

    THIS IS THE DEFECT THAT CORRUPTED graves_history.csv.

    The original implementation formatted the ``Date:`` header with strftime
    *without converting the timezone first*.  Graves sends the rack email in
    the evening, around 8 PM CT.  A header carrying a UTC offset puts that at
    01:00 the following day, so ``strftime("%Y-%m-%d")`` returned tomorrow's
    date and every affected row landed one session late: its rack price was
    then joined against the *next* day's settle.

    The fingerprint is still visible in the file -- 2023 has 42 Saturday rows
    and zero Monday rows, because a Friday evening email became Saturday and a
    Monday evening email became Tuesday.  See ``alignment.py``; those rows are
    excluded from calibration because the Monday settles needed to repair them
    were never backfilled.

    ``email.utils.parsedate_to_datetime`` handles every RFC 2822 form including
    the obsolete zone names, so the hand-rolled format list is gone too.
    """
    raw = msg.get("Date", "")
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        # A header with no zone is assumed to already be local; treating it as
        # UTC would reintroduce exactly the bug above.
        parsed = TZ.localize(parsed)
    return parsed.astimezone(TZ).date().isoformat()

def get_body(msg):
    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body_text += part.get_payload(decode=True).decode('utf-8', errors='ignore') + "\n"
            elif ctype == 'text/html' and 'attachment' not in cdispo:
                html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                body_text += BeautifulSoup(html, "html.parser").get_text(separator=" ") + "\n"
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body_text = payload.decode('utf-8', errors='ignore')
    return body_text

def extract_price_near_label(text, label):
    pattern = rf'{re.escape(label)}.*?\$?(\d+\.\d{{2,5}})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def validate_price(price, label):
    if price is None:
        return False
    if not (1.50 <= price <= 6.00):
        return False
    return True

def main():
    print("=== Graves Oil Email Extractor ===")
    email_addr = input("Gmail Address: ").strip()
    app_pw = getpass.getpass("App Password (input will be hidden): ").strip()
    
    if not email_addr or not app_pw:
        print("Error: You must provide credentials.")
        exit(1)
        
    print("\nConnecting to Gmail...")
    try:
        mail = connect_imap(email_addr, app_pw)
    except Exception as e:
        print(f"Failed to login: {e}")
        exit(1)
        
    print("Fetching Graves Oil email list...")
    email_ids = fetch_all_graves_emails(mail)
    print(f"Found {len(email_ids)} emails. Starting extraction...")
    
    # Load existing CSV if it exists
    existing_dates = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        existing_dates = set(existing_df["date"].dt.strftime("%Y-%m-%d"))
        print(f"Existing CSV has {len(existing_df)} rows.")

    new_rows = []
    errors = []
    
    for i, eid in enumerate(email_ids):
        try:
            status, msg_data = mail.fetch(eid, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            date_str = parse_email_date(msg)
            
            if not date_str:
                errors.append((eid.decode(), "Could not parse date"))
                continue
                
            if date_str in existing_dates:
                continue
                
            body = get_body(msg)
            
            row = {"date": date_str}
            valid = True
            
            for key, label in LABELS.items():
                price = extract_price_near_label(body, label)
                if not validate_price(price, label):
                    valid = False
                    errors.append((date_str, f"Missing/invalid price for {label}"))
                    break
                row[key] = price
                
            if valid:
                row["nymex_rb"] = None
                row["nymex_ho"] = None
                new_rows.append(row)
                existing_dates.add(date_str)

        except Exception as e:
            errors.append((eid.decode(), str(e)))

        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(email_ids)} emails processed...")

    mail.logout()
    
    print(f"\nExtracted {len(new_rows)} new valid rows. Errors: {len(errors)}")
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(OUTPUT_CSV):
            existing_df = pd.read_csv(OUTPUT_CSV)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values("date").reset_index(drop=True)
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        
        # Ensure column order
        cols = ["date", "rack_u", "rack_p", "rack_d", "nymex_rb", "nymex_ho"]
        for col in cols:
            if col not in combined.columns:
                combined[col] = None
        combined = combined[cols]
        
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully saved to {OUTPUT_CSV}!")

if __name__ == "__main__":
    main()
