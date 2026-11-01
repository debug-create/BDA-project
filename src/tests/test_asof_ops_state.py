#!/usr/bin/env python3
"""Synthetic leakage checks for the six-hour UTC state export."""

import os
import sys
from datetime import date

from pyspark.sql import SparkSession, types as T

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import asof_ops_state
import asof_ops_utc


EVENT_SCHEMA = T.StructType([
    T.StructField("FlightDate", T.DateType(), False),
    T.StructField("Reporting_Airline", T.StringType(), False),
    T.StructField("Origin", T.StringType(), False),
    T.StructField("Dest", T.StringType(), False),
    T.StructField("CRSDepTime", T.IntegerType(), False),
    T.StructField("CRSElapsedTime", T.DoubleType(), False),
    T.StructField("Cancelled", T.IntegerType(), False),
    T.StructField("Diverted", T.IntegerType(), False),
    T.StructField("DepDelay", T.DoubleType(), True),
    T.StructField("ArrDelay", T.DoubleType(), True),
])
PRIORS = {"dep_mean": 0.0, "dep_share": 0.1, "arr_mean": 0.0, "arr_share": 0.1,
          "carrier_backlog_share": 0.2, "smoothing": 20.0}


def timezone_row(code, day):
    return (code, day, "Etc/UTC", "MATCHED", *([0] * 48))


def compute(spark, rows, target_day=date(2024, 11, 1), target_hour=12):
    cached = []
    def keep(frame):
        frame.persist(); cached.append(frame); return frame
    def release(frame):
        try: frame.unpersist()
        except Exception: pass
    events = spark.createDataFrame(rows, EVENT_SCHEMA)
    days = sorted({row[0] for row in rows} | {target_day})
    tz = spark.createDataFrame([timezone_row(code, day) for day in days for code in ("AAA", "BBB")],
                               asof_ops_utc.timezone_schema())
    target = spark.createDataFrame(
        [("target", target_day, "AAA", "BBB", "C1", float(target_hour))],
        "FlightKey string, FlightDate date, Origin string, Dest string, Reporting_Airline string, ScheduledDepHour double")
    result, columns = asof_ops_state.attach_hourly_state_utc(
        events, target, tz, 120, PRIORS, keep, release)
    row = result.select(*columns, "RouteEarlierTodayCountLog").first().asDict()
    for frame in cached: frame.unpersist()
    return row


def test_target_outcome_is_invariant_but_observed_event_changes_state(spark):
    day = date(2024, 11, 1)
    prior = (day, "C1", "AAA", "BBB", 800, 60.0, 0, 0, 60.0, 0.0)
    target_a = (day, "C1", "AAA", "BBB", 1200, 60.0, 0, 0, 0.0, 0.0)
    target_b = (day, "C1", "AAA", "BBB", 1200, 60.0, 0, 0, 300.0, 300.0)
    first = compute(spark, [prior, target_a])
    changed_target = compute(spark, [prior, target_b])
    assert first == changed_target, "the target flight's own outcome leaked into its inputs"
    moved_after_t = (day, "C1", "AAA", "BBB", 800, 60.0, 0, 0, 120.0, 0.0)
    changed_observed = compute(spark, [moved_after_t, target_a])
    assert first["seq_origin_departures_count_h1"] > changed_observed["seq_origin_departures_count_h1"]


def test_midnight_crossing_uses_previous_utc_day(spark):
    prior_day = date(2024, 10, 31); target_day = date(2024, 11, 1)
    prior = (prior_day, "C1", "AAA", "BBB", 2100, 60.0, 0, 0, 120.0, 0.0)
    target = (target_day, "C1", "AAA", "BBB", 200, 60.0, 0, 0, 0.0, 0.0)
    values = compute(spark, [prior, target], target_day, 2)
    assert values["seq_origin_departures_count_h1"] > 0.0


def main():
    spark = (SparkSession.builder.master("local[2]").appName("BDA-Ops-V2-State-Tests")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        for test in [test_target_outcome_is_invariant_but_observed_event_changes_state,
                     test_midnight_crossing_uses_previous_utc_day]:
            test(spark); print(f"PASS {test.__name__}", flush=True)
        print("ASOF_STATE_TESTS_STATUS=PASSED", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
