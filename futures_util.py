import os
import re
from datetime import date, datetime, time, timedelta
import calendar

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

DELIVERY_MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# Business days before the mathematical LTD at which the quoted front month
# switches.  Calibrated against the data rather than assumed: for each candidate
# value, flag the implied roll sessions and measure the residual of the rack
# regression on them.  A correct value isolates the contract price gap, where
# NYMEX jumps and the rack does not follow.
#
#   early_roll_days   flagged mean|resid|   unflagged mean|resid|
#        3                  1.91 / 2.05           2.04 / 2.33     (no separation)
#        2                  4.70 / 5.03           1.91 / 2.19     (clean separation)
#        1                  0.84 / 2.33           2.09 / 2.32     (no separation)
#
# Reproduce with ``python3 -m pytest test_futures_util.py -k roll_offset``.
DEFAULT_EARLY_ROLL_DAYS = 2


def extract_schwab_active_future_symbol(response_json, prefix):
    """Return Schwab's declared active futures contract from quote metadata."""
    if not isinstance(response_json, dict):
        return None

    expected = re.compile(rf"^/{re.escape(prefix)}[FGHJKMNQUVXZ]\d{{2}}$")

    def normalize(value):
        if not isinstance(value, str):
            return None
        symbol = value.strip().upper().split(":", 1)[0]
        if not symbol.startswith("/"):
            symbol = "/" + symbol
        return symbol if expected.fullmatch(symbol) else None

    root_symbol = f"/{prefix}"
    root_entry = response_json.get(root_symbol)
    if isinstance(root_entry, dict):
        root_active = normalize((root_entry.get("reference") or {}).get("futureActiveSymbol"))
        if root_active:
            return root_active

    # Response keys are not guaranteed to preserve the request's spelling or order.
    declared = []
    for entry in response_json.values():
        if not isinstance(entry, dict):
            continue
        reference = entry.get("reference") or {}
        active = normalize(reference.get("futureActiveSymbol"))
        if active:
            declared.append(active)
    unique_declared = list(dict.fromkeys(declared))
    if len(unique_declared) == 1:
        return unique_declared[0]

    # Some responses expose only futureIsActive on individual contract references.
    active_contracts = []
    for response_symbol, entry in response_json.items():
        if not isinstance(entry, dict):
            continue
        reference = entry.get("reference") or {}
        is_active = reference.get("futureIsActive")
        if is_active is True or (isinstance(is_active, str) and is_active.lower() == "true"):
            quote = entry.get("quote") or {}
            symbol = normalize(quote.get("symbol") or entry.get("symbol") or response_symbol)
            if symbol:
                active_contracts.append(symbol)

    unique_active = list(dict.fromkeys(active_contracts))
    return unique_active[0] if len(unique_active) == 1 else None

def add_month(year, month, offset):
    month += offset
    year += (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return year, month

def is_nymex_business_day(day):
    if day.weekday() >= 5:
        return False
    if not mcal:
        return day not in us_market_holidays(day.year)
    try:
        cal = mcal.get_calendar('NYMEX')
        schedule = cal.schedule(start_date=day, end_date=day)
        return not schedule.empty
    except Exception:
        return day not in us_market_holidays(day.year)

def observed_fixed_holiday(year, month, day):
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday

def nth_weekday(year, month, weekday, n):
    day = date(year, month, 1)
    while day.weekday() != weekday:
        day += timedelta(days=1)
    return day + timedelta(days=7 * (n - 1))

def last_weekday(year, month, weekday):
    day = date(year, month, calendar.monthrange(year, month)[1])
    while day.weekday() != weekday:
        day -= timedelta(days=1)
    return day

def easter_date(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def us_market_holidays(year):
    return {
        observed_fixed_holiday(year, 1, 1),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        easter_date(year) - timedelta(days=2),
        last_weekday(year, 5, 0),
        observed_fixed_holiday(year, 6, 19),
        observed_fixed_holiday(year, 7, 4),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 11, 3, 4),
        observed_fixed_holiday(year, 12, 25),
    }

def previous_nymex_business_day(day):
    day -= timedelta(days=1)
    while not is_nymex_business_day(day):
        day -= timedelta(days=1)
    return day

def last_nymex_business_day(year, month):
    day = date(year, month, calendar.monthrange(year, month)[1])
    while not is_nymex_business_day(day):
        day -= timedelta(days=1)
    return day

def refined_product_last_trade_date(contract_year, contract_month):
    prev_year, prev_month = add_month(contract_year, contract_month, -1)
    return last_nymex_business_day(prev_year, prev_month)

def crude_last_trade_date(contract_year, contract_month):
    prev_year, prev_month = add_month(contract_year, contract_month, -1)
    twenty_fifth = date(prev_year, prev_month, 25)
    business_days = 4 if not is_nymex_business_day(twenty_fifth) else 3
    day = twenty_fifth
    for _ in range(business_days):
        day = previous_nymex_business_day(day)
    return day

def contract_last_trade_date(contract_year, contract_month, prefix):
    if prefix in ('RB', 'HO'):
        return refined_product_last_trade_date(contract_year, contract_month)
    if prefix == 'CL':
        return crude_last_trade_date(contract_year, contract_month)
    raise ValueError(f"Unsupported futures prefix: {prefix}")

def get_front_month_contract(dt, prefix, early_roll_days=None):
    """Return (contract_year, contract_month, ltd) for the active front-month contract.

    early_roll_days: number of NYMEX business days before the mathematical LTD at which
    the contract is considered rolled.  Defaults to ``DEFAULT_EARLY_ROLL_DAYS``, which is
    calibrated against observed settle gaps rather than assumed -- see the constant's
    definition.  The primary resolution (Schwab/yfinance active symbol) overrides this.
    """
    if early_roll_days is None:
        early_roll_days = DEFAULT_EARLY_ROLL_DAYS
    if isinstance(dt, datetime):
        today = dt.date()
    else:
        today = dt
    contract_year, contract_month = add_month(today.year, today.month, 1)
    for _ in range(24):
        ltd = contract_last_trade_date(contract_year, contract_month, prefix)
        # Check if today is within early_roll_days business days of the LTD.
        # Count back early_roll_days business days from the LTD to get the effective roll date.
        effective_roll_date = ltd
        days_back = 0
        candidate = ltd
        while days_back < early_roll_days:
            candidate -= timedelta(days=1)
            if is_nymex_business_day(candidate):
                days_back += 1
        effective_roll_date = candidate  # first day of the early-roll window
        if today <= effective_roll_date:
            return contract_year, contract_month, ltd
        contract_year, contract_month = add_month(contract_year, contract_month, 1)
    raise RuntimeError(f"Could not resolve front-month contract for {prefix}")

def is_contract_roll_day(dt, prefix, early_roll_days=None):
    """True when this session's front-month contract differs from the previous
    session's, so a settle-to-settle difference spans two different contracts.

    Defined as a *change* rather than as a calendar landmark.  The previous
    implementation compared ``today == ltd`` against the LTD returned by
    ``get_front_month_contract``, but that function already advances to the next
    contract ``early_roll_days`` business days before the LTD.  By the time the
    comparison ran, ``ltd`` was next month's LTD, so the equality could never
    hold and the function returned False for every date of every year -- see
    ``test_futures_util.py::test_roll_day_fires_on_contract_change``.

    Comparing consecutive resolutions is immune to that class of bug: whatever
    roll convention ``get_front_month_contract`` implements, this flags exactly
    the session on which it switches, and only that session.  The following
    session compares two settles of the same new contract and is clean.
    """
    if isinstance(dt, datetime):
        today = dt.date()
    else:
        today = dt
    try:
        # A weekend or holiday is not a session, so it cannot be a roll session.
        # Without this guard every Saturday and Sunday following a Friday roll
        # also compares against that Friday and reports True.
        if not is_nymex_business_day(today):
            return False
        prev_day = previous_nymex_business_day(today)
        today_contract = get_front_month_contract(
            today, prefix, early_roll_days=early_roll_days)[:2]
        prev_contract = get_front_month_contract(
            prev_day, prefix, early_roll_days=early_roll_days)[:2]
        return today_contract != prev_contract
    except Exception:
        return False

def get_front_month_schwab_symbol(dt, prefix):
    contract_year, contract_month, _ = get_front_month_contract(dt, prefix)
    code = DELIVERY_MONTH_CODES[contract_month]
    return f"/{prefix}{code}{contract_year % 100:02d}"
