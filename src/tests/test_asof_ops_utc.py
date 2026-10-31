#!/usr/bin/env python3
"""Synthetic UTC-ordering and DST tests for ops-v2 features."""

import os
import sys
from datetime import date

from pyspark.sql import SparkSession, types as T

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import asof_ops
import asof_ops_utc


EVENT_SCHEMA = T.StructType([
    T.StructField("FlightDate", T.DateType(), False),
    T.StructField("Reporting_Airline", T.StringType(), False),
    T.StructField("Origin", T.StringType(), False),
    T.StructField("Dest", T.StringType(), False),
    T.StructField("CRSDepTime", T.IntegerType(), False),
    T.StructField("CRSArrTime", T.IntegerType(), False),
    T.StructField("CRSElapsedTime", T.DoubleType(), False),
    T.StructField("Cancelled", T.IntegerType(), False),
    T.StructField("Diverted", T.IntegerType(), False),
    T.StructField("DepDelay", T.DoubleType(), True),
    T.StructField("ArrDelay", T.DoubleType(), True),
    T.StructField("ArrDel15", T.IntegerType(), True),
])
PRIORS = {"dep_mean": 0.0, "dep_share": 0.1, "arr_mean": 0.0, "arr_share": 0.1,
          "origin_backlog_share": 0.2, "carrier_backlog_share": 0.2, "smoothing": 20.0}


def timezone_row(code, day, zone, fold0, fold1=None):
    fold1 = fold0 if fold1 is None else fold1
    return (code, day, zone, "MATCHED", *fold0, *fold1)


def fixed_offsets(value):
    return [value] * 24


def target_frame(spark, day, hour, origin, dest):
    return spark.createDataFrame(
        [(day, origin, dest, "C1", float(hour), 1.0)],
        "FlightDate date, Origin string, Dest string, Reporting_Airline string, ScheduledDepHour double, label double")


def compute_utc(spark, rows, target, timezone_rows):
    cached = []

    def keep(frame):
        frame.persist(); cached.append(frame); return frame

    def release(frame):
        try:
            frame.unpersist()
        except Exception:
            pass
        cached[:] = [item for item in cached if item is not frame]

    events = spark.createDataFrame(rows, EVENT_SCHEMA)
    tz = spark.createDataFrame(timezone_rows, asof_ops_utc.timezone_schema())
    result, _, _ = asof_ops_utc.attach_asof_features_utc(
        events, target, 120, target.first()[0], tz, keep, release, priors=PRIORS)
    values = result.select(*asof_ops.OPS_FEATURES).first().asDict()
    for frame in cached:
        frame.unpersist()
    return values


def compute_old(spark, rows, target):
    cached = []

    def keep(frame):
        frame.persist(); cached.append(frame); return frame

    def release(frame):
        try:
            frame.unpersist()
        except Exception:
            pass
        cached[:] = [item for item in cached if item is not frame]

    events = spark.createDataFrame(rows, EVENT_SCHEMA)
    result, _ = asof_ops.attach_asof_features(
        events, target, 120, target.first()[0], keep, release, priors=PRIORS)
    values = result.select(*asof_ops.OPS_FEATURES).first().asDict()
    for frame in cached:
        frame.unpersist()
    return values


def test_westbound_post_t_event_is_excluded(spark):
    day = date(2024, 11, 1)
    target = target_frame(spark, day, 12, "JFK", "LAX")
    # Target: JFK 12:00 EDT = 16:00 UTC, so T=14:00 UTC.  The LAX event
    # lands at 09:00 PDT = 16:00 UTC: two hours after T, although 09:00 is
    # before the old target-local 10:00 wall-clock cutoff.
    event = (day, "C2", "LAX", "LAX", 700, 900, 120.0, 0, 0, 0.0, 0.0, 0)
    timezone_rows = [
        timezone_row("JFK", day, "America/New_York", fixed_offsets(-240)),
        timezone_row("LAX", day, "America/Los_Angeles", fixed_offsets(-420)),
    ]
    utc_values = compute_utc(spark, [event], target, timezone_rows)
    old_values = compute_old(spark, [event], target)
    assert utc_values["OpsDestLandedCount3h"] == 0.0
    assert old_values["OpsDestLandedCount3h"] == 1.0, "old local-clock implementation was not exposed"


def test_dst_fallback_uses_conservative_fold(spark):
    day = date(2024, 11, 3)
    target = target_frame(spark, day, 3, "JFK", "JFK")
    # At fall-back, local 01:00 can mean 05:00 or 06:00 UTC.  Events use
    # the later 06:00 interpretation; target 03:00 EST minus 2h is 06:00.
    # Same-hour events must therefore be excluded by the strict-hour rule.
    fold0 = [-240, -240] + [-300] * 22
    fold1 = [-240, -300] + [-300] * 22
    timezone_rows = [timezone_row("JFK", day, "America/New_York", fold0, fold1)]
    event = (day, "C2", "JFK", "JFK", 100, 100, 0.0, 0, 0, 0.0, 0.0, 0)
    values = compute_utc(spark, [event], target, timezone_rows)
    assert values["OpsOriginDepartedCount3h"] == 0.0
    assert values["OpsOriginInboundLandedCount3h"] == 0.0


def main():
    spark = (SparkSession.builder.master("local[2]").appName("BDA-AsOf-UTC-Tests")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    tests = [test_westbound_post_t_event_is_excluded, test_dst_fallback_uses_conservative_fold]
    try:
        for test in tests:
            test(spark)
            print(f"PASS {test.__name__}", flush=True)
        print("ASOF_UTC_TESTS_STATUS=PASSED", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
