#!/usr/bin/env python3
"""Synthetic invariance tests for conservative T-minus-lead features."""

import os
import sys
from datetime import date

from pyspark.sql import SparkSession, functions as F, types as T

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import asof_ops


SCHEMA = T.StructType([
    T.StructField("FlightDate", T.DateType(), False),
    T.StructField("Reporting_Airline", T.StringType(), False),
    T.StructField("Origin", T.StringType(), False),
    T.StructField("Dest", T.StringType(), False),
    T.StructField("CRSDepTime", T.IntegerType(), False),
    T.StructField("CRSArrTime", T.IntegerType(), False),
    T.StructField("Cancelled", T.IntegerType(), False),
    T.StructField("Diverted", T.IntegerType(), False),
    T.StructField("DepDelay", T.DoubleType(), True),
    T.StructField("ArrDelay", T.DoubleType(), True),
    T.StructField("ArrDel15", T.IntegerType(), True),
])
PRIORS = {"dep_mean": 0.0, "dep_share": 0.10, "arr_mean": 0.0, "arr_share": 0.10,
          "origin_backlog_share": 0.20, "carrier_backlog_share": 0.20, "smoothing": 20.0}


def target_frame(spark, day=date(2024, 1, 2), dep_hour=12, origin="AAA", dest="BBB"):
    return spark.createDataFrame(
        [(day, origin, dest, "C1", float(dep_hour), 1.0)],
        "FlightDate date, Origin string, Dest string, Reporting_Airline string, ScheduledDepHour double, label double")


def base_rows():
    d = date(2024, 1, 2)
    return [
        # Observed departure at 07:30, inside [T-3h,T) for target T=10:00.
        (d, "C1", "AAA", "CCC", 700, 900, 0, 0, 30.0, 10.0, 0),
        # Scheduled before T but actual departure/arrival after T: backlog only.
        (d, "C1", "AAA", "BBB", 900, 1000, 0, 0, 120.0, 120.0, 1),
        # Carrier-network departure at 06:15.
        (d, "C1", "CCC", "DDD", 600, 800, 0, 0, 15.0, 0.0, 0),
        # Arrival at target origin at 08:20.
        (d, "C2", "DDD", "AAA", 700, 800, 0, 0, 0.0, 20.0, 1),
        # Arrival at target destination at 09:00.
        (d, "C2", "DDD", "BBB", 800, 900, 0, 0, 0.0, 0.0, 0),
        # Target flight itself; its outcomes must never contribute.
        (d, "C1", "AAA", "BBB", 1200, 1400, 0, 0, 10.0, 10.0, 0),
    ]


def compute(spark, rows, target=None):
    cached = []

    def keep(frame):
        frame.persist()
        cached.append(frame)
        return frame

    def release(frame):
        try:
            frame.unpersist()
        except Exception:
            pass
        cached[:] = [item for item in cached if item is not frame]

    events = spark.createDataFrame(rows, SCHEMA)
    eligible = target if target is not None else target_frame(spark)
    result, _ = asof_ops.attach_asof_features(
        events, eligible, 120, date(2024, 1, 2), keep, release, priors=PRIORS)
    values = result.select(*asof_ops.OPS_FEATURES).first().asDict()
    for frame in cached:
        frame.unpersist()
    return values


def assert_same(left, right, message):
    assert left.keys() == right.keys()
    for key in left:
        assert abs(float(left[key]) - float(right[key])) < 1e-12, (message, key, left[key], right[key])


def test_future_outcomes_are_invariant(spark):
    rows = base_rows()
    original = compute(spark, rows)
    changed = list(rows)
    future = list(changed[1])
    future[6], future[8], future[9], future[10] = 1, 180.0, 180.0, 0
    changed[1] = tuple(future)
    mutated = compute(spark, changed)
    assert_same(original, mutated, "unobserved flight changed as-of features")

    # Deliberately leaky scheduled-hour aggregation does change, proving this
    # mutation would be caught if future DepDelay were used.
    before_leaky = sum(row[8] for row in rows if row[2] == "AAA" and row[4] < 1000)
    after_leaky = sum(row[8] for row in changed if row[2] == "AAA" and row[4] < 1000)
    assert before_leaky != after_leaky


def test_observed_departure_changes_features(spark):
    rows = base_rows()
    original = compute(spark, rows)
    changed = list(rows)
    observed = list(changed[0]); observed[8] = 60.0; changed[0] = tuple(observed)
    mutated = compute(spark, changed)
    assert original["OpsOriginMeanDepDelay3h"] != mutated["OpsOriginMeanDepDelay3h"]


def test_target_outcome_never_contributes_and_rows_preserved(spark):
    rows = base_rows()
    original = compute(spark, rows)
    changed = list(rows)
    own = list(changed[-1]); own[6], own[7], own[8], own[9], own[10] = 1, 1, -300.0, -300.0, 1
    changed[-1] = tuple(own)
    mutated = compute(spark, changed)
    assert_same(original, mutated, "target outcome changed its own features")


def test_midnight_crossing_local_arrival(spark):
    target = target_frame(spark, day=date(2024, 1, 2), dep_hour=2, origin="AAA", dest="BBB")
    # CRSArr < CRSDep means arrival is Jan 2 01:00 local. At -60 minutes it
    # falls in T's hour and is conservatively excluded; at -61 it is hour 23
    # on Jan 1 and therefore observed strictly before T's midnight hour.
    base = [(date(2024, 1, 1), "C2", "DDD", "AAA", 2300, 100, 0, 0, 0.0, -60.0, 0)]
    excluded = compute(spark, base, target)
    changed = [tuple(list(base[0])[:9] + [-61.0, 0])]
    included = compute(spark, changed, target)
    assert excluded["OpsOriginInboundLandedCount3h"] == 0.0
    assert included["OpsOriginInboundLandedCount3h"] == 1.0


def test_airport_local_clock_timezone_isolation(spark):
    """An event on another airport's local clock cannot enter AAA's state.

    The clean handoff has airport codes but no UTC offsets.  The production
    approximation therefore keeps every airport-hour table keyed by airport;
    it never pools equal wall-clock hours across airports for an airport-level
    feature.  Cross-airport network timing is documented separately.
    """
    target = target_frame(spark, origin="AAA", dest="BBB")
    day = date(2024, 1, 2)
    aaa_arrival = (day, "C2", "DDD", "AAA", 700, 800, 0, 0, 0.0, 20.0, 1)
    other_zone_arrival = (day, "C2", "DDD", "ZZZ", 700, 800, 0, 0, 0.0, 80.0, 1)
    only_aaa = compute(spark, [aaa_arrival], target)
    with_other_zone = compute(spark, [aaa_arrival, other_zone_arrival], target)
    assert_same(only_aaa, with_other_zone, "another airport's local-hour event crossed airport keys")


def main():
    spark = (SparkSession.builder.master("local[2]").appName("BDA-AsOf-Ops-Tests")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    tests = [test_future_outcomes_are_invariant, test_observed_departure_changes_features,
             test_target_outcome_never_contributes_and_rows_preserved, test_midnight_crossing_local_arrival,
             test_airport_local_clock_timezone_isolation]
    try:
        for test in tests:
            test(spark)
            print(f"PASS {test.__name__}", flush=True)
        print("ASOF_OPS_TESTS_STATUS=PASSED", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
