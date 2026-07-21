import argparse
import base64
import csv
import json
import os
from datetime import date, datetime, time, timedelta, timezone

import requests

from futures_util import DELIVERY_MONTH_CODES, add_month


API_BASE = "https://api.schwabapi.com"
PRICE_TOLERANCE = 0.00015


def refresh_access_token():
    app_key = os.environ["SCHWAB_APP_KEY"]
    app_secret = os.environ["SCHWAB_APP_SECRET"]
    old_refresh = os.environ["SCHWAB_REFRESH_TOKEN"]
    auth = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
    response = requests.post(
        f"{API_BASE}/v1/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": old_refresh},
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    new_refresh = payload.get("refresh_token")
    if new_refresh and new_refresh != old_refresh:
        update_github_refresh_secret(new_refresh)
        print("Rotated Schwab refresh token persisted to GitHub Secrets.")
    return payload["access_token"]


def update_github_refresh_secret(refresh_token):
    from nacl import encoding, public

    repo = os.environ["GH_REPO"]
    headers = {
        "Authorization": f"token {os.environ['GH_PAT']}",
        "Accept": "application/vnd.github.v3+json",
    }
    key_response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/secrets/public-key",
        headers=headers,
        timeout=20,
    )
    key_response.raise_for_status()
    key_data = key_response.json()
    public_key = public.PublicKey(
        key_data["key"].encode(), encoding.Base64Encoder
    )
    encrypted = public.SealedBox(public_key).encrypt(refresh_token.encode())
    response = requests.put(
        f"https://api.github.com/repos/{repo}/actions/secrets/SCHWAB_REFRESH_TOKEN",
        headers=headers,
        json={
            "encrypted_value": base64.b64encode(encrypted).decode(),
            "key_id": key_data["key_id"],
        },
        timeout=20,
    )
    response.raise_for_status()


def candidate_symbols(prefix, start_date, end_date):
    year, month = add_month(start_date.year, start_date.month, -1)
    final_year, final_month = add_month(end_date.year, end_date.month, 3)
    symbols = []
    while (year, month) <= (final_year, final_month):
        symbols.append(f"/{prefix}{DELIVERY_MONTH_CODES[month]}{year % 100:02d}")
        year, month = add_month(year, month, 1)
    return symbols


def fetch_daily_closes(access_token, symbol, start_date, end_date):
    start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date + timedelta(days=1), time.max, tzinfo=timezone.utc)
    response = requests.get(
        f"{API_BASE}/marketdata/v1/pricehistory",
        params={
            "symbol": symbol,
            "periodType": "year",
            "frequencyType": "daily",
            "frequency": 1,
            "startDate": int(start_dt.timestamp() * 1000),
            "endDate": int(end_dt.timestamp() * 1000),
            "needExtendedHoursData": "false",
        },
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    response.raise_for_status()
    closes = {}
    for candle in response.json().get("candles", []):
        timestamp = candle.get("datetime")
        close = candle.get("close")
        if timestamp is None or close is None:
            continue
        candle_date = datetime.fromtimestamp(
            float(timestamp) / 1000, tz=timezone.utc
        ).date()
        closes[candle_date] = float(close)
    return closes


def matching_symbols(price, prices_by_symbol, session_date, tolerance=PRICE_TOLERANCE):
    if price is None:
        return []
    return sorted(
        symbol
        for symbol, closes in prices_by_symbol.items()
        if session_date in closes
        and abs(float(closes[session_date]) - float(price)) <= tolerance
    )


def assess_observation(
    prefix,
    session_date,
    stored_price,
    continuous_price,
    prices_by_symbol,
    tolerance=PRICE_TOLERANCE,
):
    result = {
        "date": session_date.isoformat(),
        "commodity": prefix,
        "stored_price": stored_price,
        "schwab_continuous_price": continuous_price,
        "stored_matches": matching_symbols(
            stored_price, prices_by_symbol, session_date, tolerance
        ),
        "active_matches": matching_symbols(
            continuous_price, prices_by_symbol, session_date, tolerance
        ),
        "status": "unverifiable",
        "correction_eligible": False,
    }
    if continuous_price is None:
        return result
    if abs(float(stored_price) - float(continuous_price)) <= tolerance:
        result["status"] = "verified"
        return result
    result["status"] = "mismatch"
    result["correction_eligible"] = len(result["active_matches"]) == 1
    return result


def load_observations(csv_path, start_date, end_date):
    observations = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            session_date = date.fromisoformat(row["date"])
            if not start_date <= session_date <= end_date:
                continue
            for prefix, column in (("RB", "nymex_rb"), ("HO", "nymex_ho")):
                value = row.get(column)
                if value not in (None, ""):
                    observations.append((session_date, prefix, float(value)))
    return observations


def run_audit(csv_path, start_date, end_date):
    access_token = refresh_access_token()
    observations = load_observations(csv_path, start_date, end_date)
    results = []
    for prefix in ("RB", "HO"):
        root = f"/{prefix}"
        root_closes = fetch_daily_closes(access_token, root, start_date, end_date)
        contract_closes = {}
        for symbol in candidate_symbols(prefix, start_date, end_date):
            try:
                contract_closes[symbol] = fetch_daily_closes(
                    access_token, symbol, start_date, end_date
                )
            except requests.RequestException as exc:
                print(f"Skipping unavailable {symbol}: {exc}")

        for session_date, commodity, stored_price in observations:
            if commodity != prefix:
                continue
            results.append(
                assess_observation(
                    prefix,
                    session_date,
                    stored_price,
                    root_closes.get(session_date),
                    contract_closes,
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--csv", default="data/graves_history.csv")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be on or after --start")

    results = run_audit(args.csv, args.start, args.end)
    for result in results:
        print("CONTRACT_AUDIT " + json.dumps(result, sort_keys=True))
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("verified", "mismatch", "unverifiable")
    }
    eligible = sum(result["correction_eligible"] for result in results)
    print(f"CONTRACT_AUDIT_SUMMARY {json.dumps({**counts, 'correction_eligible': eligible}, sort_keys=True)}")


if __name__ == "__main__":
    main()
