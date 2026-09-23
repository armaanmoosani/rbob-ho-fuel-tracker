"""Record one physical purchasing decision against its live recommendation."""

import argparse
import csv
import os
import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTION_LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")
EXECUTION_LOG_PATH = os.path.join(DATA_DIR, "execution_log.csv")
EXECUTION_COLUMNS = [
    "record_id", "signal_date", "commodity", "recommendation", "executed_action",
    "gallons", "load_date", "price_paid_per_gallon",
    "counterfactual_price_per_gallon", "realized_savings_dollars",
    "notes", "recorded_at", "supersedes_record_id",
]


def _iso_date(value):
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def _positive_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def live_recommendation(signal_date, commodity, prediction_path=PREDICTION_LOG_PATH):
    with open(prediction_path, newline="") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if row.get("timestamp", "")[:10] == signal_date
                   and row.get("commodity") == commodity
                   and row.get("prediction_source") == "live"]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one live {commodity} prediction on {signal_date}; "
            f"found {len(matches)}")
    return matches[0]["predicted_direction"]


def write_execution(row, path=EXECUTION_LOG_PATH, supersedes=""):
    existing = []
    if os.path.exists(path):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXECUTION_COLUMNS:
                raise ValueError("execution_log.csv has an unexpected schema")
            existing = list(reader)
    superseded_ids = {item["supersedes_record_id"] for item in existing
                      if item["supersedes_record_id"]}
    active = [item for item in existing if item["record_id"] not in superseded_ids]
    duplicate = [item for item in active
                 if item["signal_date"] == row["signal_date"]
                 and item["commodity"] == row["commodity"]]
    if supersedes:
        target = next((item for item in active if item["record_id"] == supersedes), None)
        if target is None:
            raise ValueError("--supersedes must identify an active execution record")
        if (target["signal_date"], target["commodity"]) != (
                row["signal_date"], row["commodity"]):
            raise ValueError("a correction must keep the original signal date and commodity")
    elif duplicate:
        raise ValueError(
            "execution already recorded; append a correction with --supersedes RECORD_ID")
    row["supersedes_record_id"] = supersedes
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXECUTION_COLUMNS, lineterminator="\n")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record an actual fuel execution and optional invoice economics.")
    parser.add_argument("--date", required=True, type=_iso_date,
                        help="live signal date, YYYY-MM-DD")
    parser.add_argument("--commodity", required=True, choices=("RB", "HO"))
    parser.add_argument("--action", required=True,
                        choices=("DISPATCHED_SAME_DAY", "WAITED", "NO_ACTION"))
    parser.add_argument("--gallons", required=True, type=_positive_float)
    parser.add_argument("--load-date", type=_iso_date, default="")
    parser.add_argument("--price-paid", type=_positive_float)
    parser.add_argument("--counterfactual-price", type=_positive_float,
                        help="price that would have applied under the alternative action")
    parser.add_argument("--notes", default="")
    parser.add_argument("--supersedes", default="",
                        help="record_id of an immutable execution entry being corrected")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.price_paid is None) != (args.counterfactual_price is None):
        raise SystemExit("--price-paid and --counterfactual-price must be supplied together")
    if args.action == "DISPATCHED_SAME_DAY" and args.load_date != args.date:
        raise SystemExit("DISPATCHED_SAME_DAY requires --load-date equal to --date")
    if args.action == "WAITED" and args.load_date and args.load_date <= args.date:
        raise SystemExit("WAITED requires --load-date after the signal date")

    recommendation = live_recommendation(args.date, args.commodity)
    realized = ""
    if args.price_paid is not None:
        realized = (args.counterfactual_price - args.price_paid) * args.gallons
    row = {
        "record_id": uuid.uuid4().hex,
        "signal_date": args.date,
        "commodity": args.commodity,
        "recommendation": recommendation,
        "executed_action": args.action,
        "gallons": f"{args.gallons:.2f}",
        "load_date": args.load_date,
        "price_paid_per_gallon": (
            f"{args.price_paid:.4f}" if args.price_paid is not None else ""),
        "counterfactual_price_per_gallon": (
            f"{args.counterfactual_price:.4f}"
            if args.counterfactual_price is not None else ""),
        "realized_savings_dollars": f"{realized:.2f}" if realized != "" else "",
        "notes": args.notes,
        "recorded_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "supersedes_record_id": "",
    }
    write_execution(row, supersedes=args.supersedes)
    print(f"Recorded {args.date} {args.commodity}: {recommendation} -> {args.action} "
          f"(record {row['record_id']}).")
    if realized != "":
        print(f"Actual alternative-price savings: ${realized:,.2f}.")


if __name__ == "__main__":
    main()
