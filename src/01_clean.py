#!/usr/bin/env python3
"""Clean BTS Oct-Dec 2024 flight CSVs in HDFS and write Parquet."""

import json
import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, functions as F

INPUT_PATH = "hdfs://master:9000/user/root/flights/raw/flights_2024_*.csv"
OUTPUT_PATH = "hdfs://master:9000/user/root/flights/clean"
LOCAL_SUMMARY_PATH = "/results/01_clean_summary.txt"

# Keep only the columns needed for cleaning, holiday analysis, and ML.
# All CSV columns are initially read as strings to avoid slow/fragile inference.
KEEP_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime",
    "CRSElapsedTime", "Distance", "DistanceGroup", "DepTimeBlk",
    "Cancelled", "CancellationCode", "Diverted", "DepDelay",
    "DepDelayMinutes", "ArrDelay", "ArrDel15", "CarrierDelay",
    "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]

INTEGER_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "CRSDepTime", "CRSArrTime", "DistanceGroup", "Cancelled", "Diverted",
    "ArrDel15",
]

DOUBLE_COLUMNS = [
    "CRSElapsedTime", "Distance", "DepDelay", "DepDelayMinutes",
    "ArrDelay", "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]


def main() -> int:
    spark = (
        SparkSession.builder
        .appName("BDA-01-Clean-Flights")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Input HDFS path: {INPUT_PATH}", flush=True)
        raw = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .option("mode", "PERMISSIVE")
            .csv(INPUT_PATH)
        )

        # BTS monthly headers can end in a trailing comma, creating _c109.
        if "_c109" in raw.columns:
            raw = raw.drop("_c109")
            print("Dropped trailing empty header column: _c109", flush=True)

        missing_columns = [name for name in KEEP_COLUMNS if name not in raw.columns]
        if missing_columns:
            print("ERROR: required source columns were not found:", missing_columns, flush=True)
            print("First source columns:", raw.columns[:30], flush=True)
            return 2

        raw_rows = raw.count()
        print(f"Rows read from HDFS: {raw_rows:,}", flush=True)
        print(f"Source column count after optional _c109 removal: {len(raw.columns)}", flush=True)

        # Normalize empty/whitespace-only strings to null before casting.
        selected = raw.select(*[
            F.when(F.trim(F.col(name)) == "", F.lit(None).cast("string"))
             .otherwise(F.trim(F.col(name)))
             .alias(name)
            for name in KEEP_COLUMNS
        ])

        typed = selected.withColumn(
            "FlightDate", F.to_date(F.col("FlightDate"), "yyyy-MM-dd")
        )

        # Some BTS numeric flags are written as values such as 0.00 / 1.00.
        for name in INTEGER_COLUMNS:
            typed = typed.withColumn(name, F.col(name).cast("double").cast("int"))
        for name in DOUBLE_COLUMNS:
            typed = typed.withColumn(name, F.col(name).cast("double"))

        # Only drop rows missing essential identity/date fields. Keep cancelled
        # and diverted flights for descriptive analysis; ML filters them later.
        valid_condition = (
            F.col("FlightDate").isNotNull()
            & F.col("Reporting_Airline").isNotNull()
            & (F.length(F.col("Reporting_Airline")) > 0)
            & F.col("Origin").isNotNull()
            & (F.length(F.col("Origin")) > 0)
            & F.col("Dest").isNotNull()
            & (F.length(F.col("Dest")) > 0)
        )
        cleaned = typed.filter(valid_condition)

        # Get row count and null profile with one aggregate action.
        aggregate_expressions = [F.count(F.lit(1)).alias("cleaned_rows")]
        aggregate_expressions.extend(
            F.sum(F.col(name).isNull().cast("long")).alias(f"null_{name}")
            for name in KEEP_COLUMNS
        )
        profile = cleaned.agg(*aggregate_expressions).first().asDict()
        cleaned_rows = int(profile["cleaned_rows"])
        dropped_rows = raw_rows - cleaned_rows

        print(f"Rows retained: {cleaned_rows:,}", flush=True)
        print(f"Rows dropped for invalid/missing date or core route fields: {dropped_rows:,}", flush=True)
        print("Null counts after casting:", flush=True)
        null_counts = {}
        for name in KEEP_COLUMNS:
            value = int(profile.get(f"null_{name}") or 0)
            null_counts[name] = value
            print(f"  {name}: {value:,}", flush=True)

        print(f"Writing cleaned Parquet to: {OUTPUT_PATH}", flush=True)
        cleaned.write.mode("overwrite").parquet(OUTPUT_PATH)
        print("Parquet write completed.", flush=True)

        summary = {
            "job": "BDA-01-Clean-Flights",
            "run_time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_path": INPUT_PATH,
            "output_path": OUTPUT_PATH,
            "raw_rows": raw_rows,
            "retained_rows": cleaned_rows,
            "dropped_rows": dropped_rows,
            "source_column_count_after_optional_trailing_column_drop": len(raw.columns),
            "retained_columns": KEEP_COLUMNS,
            "null_counts_after_casting": null_counts,
            "notes": [
                "All CSV columns were initially read as strings; schema inference was disabled.",
                "Dropped _c109 if present because the BTS header may end with a trailing comma.",
                "Cancelled and diverted flights were retained for descriptive analysis.",
                "Post-departure outcome/delay-cause fields are retained for descriptive analysis only, not ML features.",
            ],
        }
        try:
            os.makedirs(os.path.dirname(LOCAL_SUMMARY_PATH), exist_ok=True)
            with open(LOCAL_SUMMARY_PATH, "w", encoding="utf-8") as output_file:
                output_file.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            print(f"Cleaning summary saved to {LOCAL_SUMMARY_PATH}", flush=True)
        except OSError as error:
            # A mounted host summary is convenient but should not invalidate
            # a successful HDFS Parquet write if it is unavailable.
            print(f"WARNING: could not save local summary: {error}", flush=True)

        print("CLEANING_JOB_STATUS=SUCCEEDED", flush=True)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
