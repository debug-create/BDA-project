#!/usr/bin/env python3
"""Small regression test for past-only historical-rate features in 03_ml.py.

Run inside the supplied Docker image with:
  spark-submit --master 'local[2]' \
    --conf spark.eventLog.enabled=false \
    --conf spark.ui.enabled=false \
    /src/tests/test_temporal_history_features.py

No BTS CSV files or HDFS dataset are required. This tests only the helper used to
construct training-row historical features; the full pipeline must still be run
on the project cluster after the patch is reviewed.
"""
from datetime import date
import importlib.util
import os

from pyspark.sql import SparkSession, functions as F, types as T

SCRIPT_PATH = os.environ.get("ML_SCRIPT_PATH", "/src/03_ml.py")


def load_ml_module(path):
    spec = importlib.util.spec_from_file_location("bda_ml_script_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ML script at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rows(day2_label=0.0, day3_label=0.0):
    # Same airline/route/time block across all days, so the historical rates
    # are expected to change only when earlier-day labels change.
    records = [
        ("d1-a", date(2024, 10, 1), 1.0, "AA", "AAA", "BBB", "0600-0659"),
        ("d1-b", date(2024, 10, 1), 1.0, "AA", "AAA", "BBB", "0600-0659"),
        ("d2-a", date(2024, 10, 2), day2_label, "AA", "AAA", "BBB", "0600-0659"),
        ("d2-b", date(2024, 10, 2), day2_label, "AA", "AAA", "BBB", "0600-0659"),
        ("d3-a", date(2024, 10, 3), day3_label, "AA", "AAA", "BBB", "0600-0659"),
        ("d3-b", date(2024, 10, 3), day3_label, "AA", "AAA", "BBB", "0600-0659"),
    ]
    schema = T.StructType([
        T.StructField("flight_id", T.StringType(), False),
        T.StructField("FlightDate", T.DateType(), False),
        T.StructField("label", T.DoubleType(), False),
        T.StructField("Reporting_Airline", T.StringType(), False),
        T.StructField("Origin", T.StringType(), False),
        T.StructField("Dest", T.StringType(), False),
        T.StructField("DepTimeBlk", T.StringType(), False),
    ])
    return spark.createDataFrame(records, schema)


def by_id(df, feature_cols):
    return {
        row["flight_id"]: tuple(float(row[c]) for c in feature_cols)
        for row in df.select("flight_id", *feature_cols).collect()
    }


def close_tuple(a, b, tol=1e-12):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("BDA-ML-Temporal-History-Regression-Test")
             .master("local[2]")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        ml = load_ml_module(SCRIPT_PATH)
        history_features = list(ml.HISTORY_KEYS.keys())

        original_input = make_rows(day2_label=0.0, day3_label=0.0)
        changed_same_and_future_input = make_rows(day2_label=1.0, day3_label=1.0)

        original = ml.apply_history_training_past_only(original_input)
        changed = ml.apply_history_training_past_only(changed_same_and_future_input)

        if original.count() != original_input.count():
            raise AssertionError("History feature joins changed the row count")
        if original.select("flight_id").distinct().count() != original.count():
            raise AssertionError("History feature joins duplicated flight IDs")

        orig_map = by_id(original, history_features)
        changed_map = by_id(changed, history_features)

        # Day 1 has no prior observations; the documented neutral fallback is 0.5.
        for flight_id in ("d1-a", "d1-b"):
            if not all(abs(value - 0.5) <= 1e-12 for value in orig_map[flight_id]):
                raise AssertionError(f"Expected 0.5 neutral prior on first date for {flight_id}: {orig_map[flight_id]}")

        # Changing labels on day 2 itself and day 3 must not change day-2 features.
        for flight_id in ("d2-a", "d2-b"):
            if not close_tuple(orig_map[flight_id], changed_map[flight_id]):
                raise AssertionError(f"Same-day/future labels changed history features for {flight_id}")

        # Earlier labels should influence a later date: on day 3, the altered day-2
        # labels are now legitimately historical and should affect at least one rate.
        if close_tuple(orig_map["d3-a"], changed_map["d3-a"]):
            raise AssertionError("Changing prior-day labels did not affect day-3 history rates")

        print("TEMPORAL_HISTORY_TEST_STATUS=PASSED", flush=True)
        print("Assertions: first-day fallback; same-day exclusion; future-day exclusion; prior-day responsiveness; row-count preservation.", flush=True)
    finally:
        spark.stop()
