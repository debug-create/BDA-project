#!/usr/bin/env python3
"""Build a date/hour-aware IATA-to-UTC-offset table using airportsdata + zoneinfo."""

import argparse
import csv
import glob
import importlib.metadata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import airportsdata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", default="results/experiments/ml_redesign/ops_v2/airport_codes")
    parser.add_argument("--output", default="results/experiments/ml_redesign/asof_ops/airport_tz.csv")
    parser.add_argument("--unmatched", default="results/experiments/ml_redesign/ops_v2/airport_tz_unmatched.csv")
    parser.add_argument("--start", default="2024-09-25")
    parser.add_argument("--end", default="2025-01-05")
    return parser.parse_args()


def read_codes(path):
    parts = glob.glob(str(Path(path) / "part-*.csv"))
    if not parts:
        raise SystemExit(f"No airport-code part file under {path}")
    result = set()
    for part in parts:
        with open(part, newline="", encoding="utf-8") as handle:
            result.update(row["Airport"].strip().upper() for row in csv.DictReader(handle) if row["Airport"].strip())
    return sorted(result)


def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def offsets_for_hour(zone, day, hour):
    naive = datetime(day.year, day.month, day.day, hour)
    values = []
    for fold in (0, 1):
        offset = naive.replace(tzinfo=zone, fold=fold).utcoffset()
        values.append(int(offset.total_seconds() // 60))
    return values


def main():
    args = parse_args()
    codes = read_codes(args.codes)
    airports = airportsdata.load("IATA")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    output, unmatched_path = Path(args.output), Path(args.unmatched)
    output.parent.mkdir(parents=True, exist_ok=True)
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    offset_columns = [f"OffsetFold{fold}H{hour:02d}" for fold in (0, 1) for hour in range(24)]
    header = ["Airport", "LocalDate", "TimeZone", "MatchStatus", *offset_columns]
    unmatched = []
    rows = 0
    transition_rows = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for code in codes:
            record = airports.get(code)
            zone_name = record.get("tz") if record else None
            try:
                zone = ZoneInfo(zone_name) if zone_name else None
            except ZoneInfoNotFoundError:
                zone = None
            if zone is None:
                reason = "IATA_NOT_FOUND" if record is None else "TIMEZONE_NOT_FOUND"
                unmatched.append([code, reason, zone_name or "", "UTC aggregates use zero observed counts and train priors"])
                continue
            for day in date_range(start, end):
                fold0, fold1 = [], []
                for hour in range(24):
                    offsets = offsets_for_hour(zone, day, hour)
                    fold0.append(offsets[0]); fold1.append(offsets[1])
                if fold0 != fold1 or len(set(fold0 + fold1)) > 1:
                    transition_rows += 1
                writer.writerow([code, day.isoformat(), zone_name, "MATCHED", *fold0, *fold1])
                rows += 1
    with unmatched_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Airport", "Reason", "TimeZone", "Treatment"])
        writer.writerows(unmatched)
    version = importlib.metadata.version("airportsdata")
    print(f"AIRPORT_TZ_STATUS=SUCCEEDED airportsdata={version} airports={len(codes)} matched={len(codes)-len(unmatched)} "
          f"unmatched={len(unmatched)} rows={rows} transition_rows={transition_rows} output={output}")


if __name__ == "__main__":
    main()
