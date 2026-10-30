#!/usr/bin/env python3
"""Validate the cleaned flight Parquet and holiday calendar with Spark.

This job is intentionally read-only with respect to its inputs. It writes a
small JSON report and daily-count CSV to a host-mounted output directory.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, functions as F, types as T


DEFAULT_CLEAN_PATH = "hdfs://master:9000/user/root/flights/clean"
DEFAULT_HOLIDAYS_PATH = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
DEFAULT_OUTPUT_DIR = "/results/data_validation"

REQUIRED_COLUMNS = {
    "FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime",
    "CRSArrTime", "CRSElapsedTime", "Distance", "DepTimeBlk", "Cancelled",
    "Diverted", "ArrDel15",
}

HOLIDAY_SCHEMA = T.StructType([
    T.StructField("FlightDate", T.StringType(), True),
    T.StructField("HolidayPeriod", T.StringType(), True),
])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate cleaned BTS flight data")
    parser.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--holidays-path", default=DEFAULT_HOLIDAYS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None, spark=None) -> int:
    args = parse_args(argv)
    owns_spark = spark is None
    if spark is None:
        spark = SparkSession.builder.appName("BDA-00-Validate-Actual-Data").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Reading cleaned Parquet: {args.clean_path}", flush=True)
        flights = spark.read.parquet(args.clean_path)
        missing = sorted(REQUIRED_COLUMNS - set(flights.columns))
        if missing:
            raise ValueError(f"Cleaned Parquet is missing required columns: {missing}")

        print("Cleaned schema:", flush=True)
        flights.printSchema()

        aggregate = flights.agg(
            F.count(F.lit(1)).alias("rows"),
            F.min("FlightDate").alias("min_date"),
            F.max("FlightDate").alias("max_date"),
            F.countDistinct("FlightDate").alias("distinct_dates"),
            F.countDistinct(F.xxhash64(*[F.col(c) for c in flights.columns])).alias("distinct_row_hashes"),
            *[F.sum(F.col(c).isNull().cast("long")).alias(f"null__{c}") for c in flights.columns],
        ).first().asDict()

        row_count = int(aggregate["rows"])
        distinct_hashes = int(aggregate["distinct_row_hashes"])
        null_counts = {c: int(aggregate[f"null__{c}"] or 0) for c in flights.columns}

        eligibility = (
            flights.withColumn(
                "category",
                F.when(F.col("Cancelled").isNull() | F.col("Diverted").isNull(), "missing_flag")
                .when(F.col("Cancelled") == 1, "cancelled")
                .when(F.col("Diverted") == 1, "diverted")
                .when(F.col("ArrDel15").isNull(), "missing_target")
                .when(~F.col("ArrDel15").isin(0, 1), "invalid_target")
                .when((F.col("Cancelled") == 0) & (F.col("Diverted") == 0), "eligible")
                .otherwise("other_flag_value"),
            )
            .groupBy("category").count()
        )
        eligibility_counts = {r["category"]: int(r["count"]) for r in eligibility.collect()}
        if sum(eligibility_counts.values()) != row_count:
            raise ValueError("Eligibility categories do not reconcile to the input row count")

        daily = flights.groupBy("FlightDate").count().orderBy("FlightDate")
        daily_stats = daily.agg(F.min("count").alias("minimum"), F.max("count").alias("maximum")).first()

        print(f"Reading holiday calendar: {args.holidays_path}", flush=True)
        holidays = (
            spark.read.option("header", "true").schema(HOLIDAY_SCHEMA).csv(args.holidays_path)
            .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
        )
        holiday_rows = holidays.count()
        invalid_holiday_rows = holidays.filter(
            F.col("FlightDate").isNull() | F.col("HolidayPeriod").isNull()
        ).count()
        duplicate_holiday_dates = (
            holidays.groupBy("FlightDate").count().filter(F.col("count") > 1).count()
        )
        if invalid_holiday_rows or duplicate_holiday_dates:
            raise ValueError(
                f"Holiday calendar invalid rows={invalid_holiday_rows}, "
                f"duplicate dates={duplicate_holiday_dates}"
            )

        joined_rows = flights.join(F.broadcast(holidays), "FlightDate", "left").count()
        matched_flights = flights.join(F.broadcast(holidays), "FlightDate", "inner").count()
        calendar_dates_outside_data = (
            holidays.join(flights.select("FlightDate").distinct(), "FlightDate", "left_anti").count()
        )
        if joined_rows != row_count:
            raise ValueError(f"Holiday join changed row count from {row_count} to {joined_rows}")

        report = {
            "status": "PASSED",
            "validated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "spark_version": spark.version,
            "clean_path": args.clean_path,
            "holidays_path": args.holidays_path,
            "schema": flights.schema.jsonValue(),
            "columns": flights.columns,
            "row_count": row_count,
            "distinct_row_hashes": distinct_hashes,
            "duplicate_rows_detected_by_full_row_hash": row_count - distinct_hashes,
            "date_min": aggregate["min_date"].isoformat(),
            "date_max": aggregate["max_date"].isoformat(),
            "distinct_dates": int(aggregate["distinct_dates"]),
            "minimum_daily_rows": int(daily_stats["minimum"]),
            "maximum_daily_rows": int(daily_stats["maximum"]),
            "null_counts": null_counts,
            "eligibility_counts": eligibility_counts,
            "holiday_calendar_rows": holiday_rows,
            "holiday_calendar_duplicate_dates": duplicate_holiday_dates,
            "holiday_calendar_invalid_rows": invalid_holiday_rows,
            "calendar_dates_outside_flight_data": calendar_dates_outside_data,
            "holiday_join_rows": joined_rows,
            "flights_on_holiday_calendar_dates": matched_flights,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, "data_validation_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

        daily_path = os.path.join(args.output_dir, "daily_row_counts.csv")
        with open(daily_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["FlightDate", "Rows"])
            for row in daily.collect():
                writer.writerow([row["FlightDate"].isoformat(), int(row["count"])])

        print(json.dumps(report, indent=2), flush=True)
        print(f"DATA_VALIDATION_REPORT={report_path}", flush=True)
        print("DATA_VALIDATION_STATUS=PASSED", flush=True)
        return 0
    except Exception as error:
        print(f"DATA_VALIDATION_STATUS=FAILED: {type(error).__name__}: {error}", flush=True)
        return 1
    finally:
        if owns_spark:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
