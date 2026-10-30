#!/usr/bin/env python3
"""SYNTHETIC / LOCAL regression tests for the redesigned src/03_ml.py.

These tests run Spark in local mode on small synthetic data. They check code-level
properties (leakage-sensitive features, row preservation, splits, output interface,
that the test window cannot influence decisions). They DO NOT validate model quality
or performance on the real BTS data, and they are not the assessed YARN run.

Run inside the project image (or any Spark 3.5 + Python 3 environment):

  spark-submit --master "local[2]" --conf spark.eventLog.enabled=false \
      --conf spark.ui.enabled=false /src/tests/test_ml_pipeline.py

Optional: ML_SCRIPT_PATH=/path/to/03_ml.py (default /src/03_ml.py).
Final marker on success: ML_PIPELINE_TESTS_STATUS=PASSED
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from datetime import date, timedelta

from pyspark.sql import SparkSession, functions as F, types as T

SCRIPT_PATH = os.environ.get("ML_SCRIPT_PATH", "/src/03_ml.py")
spark = None
ml = None


def load_ml_module(path):
    spec = importlib.util.spec_from_file_location("bda_ml_script_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
CLEAN_SCHEMA = T.StructType([
    T.StructField("FlightDate", T.DateType(), False),
    T.StructField("Cancelled", T.IntegerType(), True),
    T.StructField("Diverted", T.IntegerType(), True),
    T.StructField("ArrDel15", T.IntegerType(), True),
    T.StructField("CRSDepTime", T.IntegerType(), True),
    T.StructField("CRSArrTime", T.IntegerType(), True),
    T.StructField("CRSElapsedTime", T.DoubleType(), True),
    T.StructField("Distance", T.DoubleType(), True),
    T.StructField("Reporting_Airline", T.StringType(), True),
    T.StructField("DayOfWeek", T.IntegerType(), True),
    T.StructField("Origin", T.StringType(), True),
    T.StructField("Dest", T.StringType(), True),
    T.StructField("DepTimeBlk", T.StringType(), True),
])


def synthetic_flights(days=40, per_day=60, seed=7, start=date(2024, 10, 1), flip_from=None):
    """Clean-schema rows. If flip_from is given, ArrDel15 of eligible rows on/after it is inverted
    (the random stream is identical either way, so only those labels differ)."""
    rnd = random.Random(seed)
    carriers = ["AA", "DL", "UA", "WN"]
    airports = ["ATL", "DFW", "ORD", "DEN", "LAX", "JFK"]
    times = [555, 630, 659, 800, 1245, 1700, 1930, 2359, 2400, 5]
    rows = []
    for d in range(days):
        day = start + timedelta(days=d)
        for _ in range(per_day):
            carrier = rnd.choice(carriers)
            origin, dest = rnd.sample(airports, 2)
            dep = rnd.choice(times)
            arr = (dep + 200) % 2400
            roll = rnd.random()
            cancelled, diverted, label = 0, 0, None
            p = 0.10 + 0.08 * (carrier == "WN") + 0.06 * (dep >= 1700) + 0.0015 * d + 0.05 * (origin == "ORD")
            outcome = int(rnd.random() < p)
            if roll < 0.03:
                cancelled = 1
            elif roll < 0.04:
                diverted = 1
            else:
                label = outcome
                if flip_from is not None and day >= flip_from:
                    label = 1 - label
            blk_hour = min(dep // 100, 23)
            rows.append((day, cancelled, diverted, label, dep, arr, float(rnd.choice([90, 120, 180])),
                         float(rnd.choice([400, 800, 1200])), carrier, day.isoweekday(), origin, dest,
                         f"{blk_hour:02d}00-{blk_hour:02d}59"))
    return spark.createDataFrame(rows, CLEAN_SCHEMA)


def write_inputs(root, flip_from=None):
    os.makedirs(root, exist_ok=True)
    clean = os.path.join(root, "clean")
    synthetic_flights(flip_from=flip_from).write.mode("overwrite").parquet(clean)
    holidays = os.path.join(root, "holidays.csv")
    with open(holidays, "w", encoding="utf-8") as handle:
        handle.write("FlightDate,HolidayPeriod\n2024-10-10,TrainHoliday\n2024-10-26,ValidationHoliday\n"
                     "2024-11-05,TestHoliday\n")
    return clean, holidays


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


# --------------------------------------------------------------------------
# History-feature property checker (also used on a deliberately leaky function)
# --------------------------------------------------------------------------
def history_fixture(days=12, per_day=30, seed=3):
    rnd = random.Random(seed)
    rows = []
    n = 0
    for d in range(days):
        for _ in range(per_day):
            n += 1
            rows.append((f"r{n}", date(2024, 10, 1) + timedelta(days=d), float(rnd.random() < 0.15 + 0.02 * (d % 5)),
                         rnd.choice(["AA", "DL", "UA"]), rnd.choice(["ATL", "DFW", "ORD", "DEN"]),
                         rnd.choice(["JFK", "LAX", "SEA", "BOS"]), rnd.choice(["0600-0659", "1700-1759", "2100-2159"])))
    schema = T.StructType([
        T.StructField("flight_id", T.StringType(), False), T.StructField("FlightDate", T.DateType(), False),
        T.StructField("label", T.DoubleType(), False), T.StructField("Reporting_Airline", T.StringType(), False),
        T.StructField("Origin", T.StringType(), False), T.StructField("Dest", T.StringType(), False),
        T.StructField("DepTimeBlk", T.StringType(), False)])
    return spark.createDataFrame(rows, schema)


def feature_map(df, cols):
    return {r["flight_id"]: (r["FlightDate"], tuple(float(r[c]) for c in cols))
            for r in df.select("flight_id", "FlightDate", *cols).collect()}


def future_dependence_violations(history_fn, cols, base, test_dates):
    """Flip every label dated on/after D; features of rows dated <= D must not change."""
    reference = feature_map(history_fn(base), cols)
    violations = []
    for d in test_dates:
        flipped = base.withColumn("label", F.when(F.col("FlightDate") >= F.lit(d), 1.0 - F.col("label"))
                                  .otherwise(F.col("label")))
        changed = feature_map(history_fn(flipped), cols)
        for fid, (fdate, values) in reference.items():
            if fdate <= d and any(abs(a - b) > 1e-12 for a, b in zip(values, changed[fid][1])):
                violations.append((d, fid))
                break
    return violations


def leaky_leave_one_day_out(fit_base):
    """The earlier, flawed design: rate over ALL fitting dates minus the row's own date."""
    tot = fit_base.groupBy("Reporting_Airline").agg(F.sum("label").alias("tp"), F.count(F.lit(1)).alias("tn"))
    day = fit_base.groupBy("Reporting_Airline", "FlightDate").agg(F.sum("label").alias("dp"), F.count(F.lit(1)).alias("dn"))
    j = day.join(tot, "Reporting_Airline").withColumn(
        "hist_rate_airline", (F.col("tp") - F.col("dp") + F.lit(50.0)) / (F.col("tn") - F.col("dn") + F.lit(100.0)))
    return fit_base.join(j.select("Reporting_Airline", "FlightDate", "hist_rate_airline"),
                         ["Reporting_Airline", "FlightDate"], "left")


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------
def test_minute_of_day():
    rows = [(0,), (5,), (600,), (1745,), (2359,), (2400,), (1260,), (2500,), (-5,), (None,)]
    df = spark.createDataFrame(rows, "t int").withColumn("m", ml.minute_of_day(F.col("t")))
    got = {r["t"]: r["m"] for r in df.collect()}
    assert got[0] == 0 and got[5] == 5 and got[600] == 360 and got[1745] == 1065, got
    assert got[2359] == 1439 and got[2400] == 0, "2400 must wrap to minute 0"
    assert got[1260] is None, "minutes >= 60 are invalid"
    assert got[2500] is None and got[-5] is None and got[None] is None, got


def test_windows_match_baseline_split():
    dates = [date(2024, 10, 1) + timedelta(days=i) for i in range(92)]
    w = ml.compute_windows(dates)
    assert w["train"] == (date(2024, 10, 1), date(2024, 11, 24)), w
    assert w["validation"] == (date(2024, 11, 25), date(2024, 12, 3)), w
    assert w["test"] == (date(2024, 12, 4), date(2024, 12, 31)), w
    try:
        ml.compute_windows(dates[:2])
        raise AssertionError("expected PipelineError for fewer than three dates")
    except ml.PipelineError:
        pass


def test_validate_schema():
    assert ml.validate_schema(CLEAN_SCHEMA) == []
    fields = [f for f in CLEAN_SCHEMA.fields if f.name not in ("Distance", "Origin")]
    fields.append(T.StructField("Origin", T.IntegerType(), True))      # wrong type, and Distance missing
    problems = " | ".join(ml.validate_schema(T.StructType(fields)))
    assert "missing column Distance" in problems, problems
    assert "column Origin has type int" in problems, problems


def test_eligibility_categories_are_exclusive_and_sum():
    rows = [
        (date(2024, 10, 1), 0, 0, 0), (date(2024, 10, 1), 0, 0, 1), (date(2024, 10, 1), 1, 0, None),
        (date(2024, 10, 1), 0, 1, None), (date(2024, 10, 1), 0, 0, None), (date(2024, 10, 1), 0, 0, 7),
        (date(2024, 10, 1), None, 0, 0), (date(2024, 10, 1), 2, 0, 0), (date(2024, 10, 1), 1, 1, 0),
    ]
    df = spark.createDataFrame(rows, "FlightDate date, Cancelled int, Diverted int, ArrDel15 int")
    counts = {r["Category"]: r["count"] for r in df.groupBy(ml.eligibility_category_expr().alias("Category")).count().collect()}
    assert sum(counts.values()) == len(rows), counts
    assert counts == {"eligible": 2, "cancelled": 2, "diverted": 1, "missing_target": 1, "invalid_target": 1,
                      "missing_cancelled_or_diverted_flag": 1, "other_flag_value": 1}, counts
    assert df.filter(ml.eligible_condition()).count() == counts["eligible"]


def test_leakage_guard_and_provenance():
    ml.assert_no_leakage()
    for forbidden in ("ArrDelay", "DepDelay", "Tail_Number", "TaxiOut", "DivAirportLandings", "Cancelled"):
        try:
            ml.assert_no_leakage(ml.FEATURE_SOURCE_COLUMNS + [forbidden])
            raise AssertionError(f"guard did not reject {forbidden}")
        except ml.PipelineError:
            pass
    allowed = set(ml.FEATURE_SOURCE_COLUMNS)
    for spec in ml.EXPERIMENT_SPECS + [ml.combine_specs(ml.EXPERIMENT_SPECS[0], ml.EXPERIMENT_SPECS[3:5])]:
        impute, plain, history, categorical = ml.spec_columns(spec)
        for feature in impute + plain + history + categorical:
            assert feature in ml.FEATURE_PROVENANCE, f"no provenance for {feature}"
            assert set(ml.FEATURE_PROVENANCE[feature]) <= allowed, (feature, ml.FEATURE_PROVENANCE[feature])
            assert not (set(ml.FEATURE_PROVENANCE[feature]) & ml.FORBIDDEN_SOURCE_COLUMNS), feature
    # The frame fed to the models carries only allow-listed columns plus label / ordering key.
    clean = synthetic_flights(days=5, per_day=20)
    holidays = spark.createDataFrame([(date(2024, 10, 2), "H")], "FlightDate date, HolidayPeriod string")
    frame = ml.build_eligible_frame(clean, holidays, ml.compute_demand_table(clean))
    leaked = set(frame.columns) & (ml.FORBIDDEN_SOURCE_COLUMNS | {"ArrDel15"})
    assert not leaked, f"outcome columns present in model frame: {leaked}"


def test_history_training_past_only_property():
    base = history_fixture()
    all_dates = [date(2024, 10, 1) + timedelta(days=i) for i in (0, 1, 3, 6, 9, 10)]
    for window in (None, 3):
        fn = lambda df, w=window: ml.apply_history_training_past_only(df, w)
        n_in = base.count()
        out = fn(base)
        assert out.count() == n_in, "history join changed the row count"
        assert out.select("flight_id").distinct().count() == n_in, "history join duplicated flight ids"
        v = future_dependence_violations(fn, ml.HISTORY_FEATURES, base, all_dates)
        assert not v, f"window={window}: labels on/after D changed features on/before D: {v[:3]}"
        # prior-day responsiveness: flipping day-3 labels must change some later-day feature
        a = feature_map(fn(base), ml.HISTORY_FEATURES)
        flipped = base.withColumn("label", F.when(F.col("FlightDate") == F.lit(date(2024, 10, 4)), 1.0 - F.col("label"))
                                  .otherwise(F.col("label")))
        b = feature_map(fn(flipped), ml.HISTORY_FEATURES)
        later_changed = [f for f, (d, vals) in a.items() if d == date(2024, 10, 5) and vals != b[f][1]]
        assert later_changed, f"window={window}: prior-day labels had no effect on the next day"


def test_property_check_detects_the_old_leaky_design():
    base = history_fixture()
    leaky_dates = [date(2024, 10, 1) + timedelta(days=i) for i in (1, 3, 6)]
    v = future_dependence_violations(leaky_leave_one_day_out, ["hist_rate_airline"], base, leaky_dates)
    assert v, "the property test failed to detect the leave-one-day-out leakage it exists to catch"


def test_history_key_isolation_and_first_day_fallback():
    rows = []
    for d in range(1, 4):
        for k in range(4):
            rows.append((f"a{d}{k}", date(2024, 10, d), 1.0, "AA", "AAA", "BBB", "0600-0659"))
            rows.append((f"b{d}{k}", date(2024, 10, d), 0.0 if d != 1 else 1.0, "BB", "CCC", "DDD", "0700-0759"))
    schema = "flight_id string, FlightDate date, label double, Reporting_Airline string, Origin string, Dest string, DepTimeBlk string"
    base = spark.createDataFrame(rows, schema)
    out = {r["flight_id"]: r for r in ml.apply_history_training_past_only(base).collect()}
    assert abs(out["a10"]["hist_rate_airline"] - 0.5) < 1e-12, "first date must use the neutral 0.5 fallback"
    changed = base.withColumn("label", F.when((F.col("Reporting_Airline") == "BB") & (F.col("FlightDate") == F.lit(date(2024, 10, 1))), 0.0)
                              .otherwise(F.col("label")))
    out2 = {r["flight_id"]: r for r in ml.apply_history_training_past_only(changed).collect()}
    # BB's day-1 labels changed. AA's rates can move only through the shared global prior
    # (0.5 -> hand-computed 0.519 vs 1.0), while BB's own rate also loses its own history.
    # Hand-computed day-2 airline rates: BB 1.0 -> 0.4808 (delta 0.519); AA 1.0 -> 0.5192 (delta 0.481).
    for feature in ("hist_rate_airline", "hist_rate_origin"):
        d_bb = abs(out2["b20"][feature] - out["b20"][feature])
        d_aa = abs(out2["a20"][feature] - out["a20"][feature])
        assert d_bb > d_aa, f"{feature}: keys are not isolated (BB delta {d_bb:.4f} <= AA delta {d_aa:.4f})"
    assert abs(out2["b20"]["hist_rate_airline"] - (0 + 100 * 0.5) / (4 + 100)) < 1e-9


def test_trailing_window_exact_values():
    rows = []
    labels = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for d, lab in labels.items():
        for k in range(10):
            rows.append((f"r{d}_{k}", date(2024, 10, d), lab, "AA", "AAA", "BBB", "0600-0659"))
    base = spark.createDataFrame(rows, "flight_id string, FlightDate date, label double, "
                                 "Reporting_Airline string, Origin string, Dest string, DepTimeBlk string")
    S = ml.HISTORY_SMOOTHING

    def rate(df, d):
        return df.filter(F.col("FlightDate") == F.lit(date(2024, 10, d))).select("hist_rate_airline").first()[0]
    expanding = ml.apply_history_training_past_only(base, None)
    trailing2 = ml.apply_history_training_past_only(base, 2)
    assert abs(rate(expanding, 4) - (10 + S * (10 / 30)) / (30 + S)) < 1e-9, rate(expanding, 4)
    assert abs(rate(trailing2, 4) - (0 + S * 0.0) / (20 + S)) < 1e-9, rate(trailing2, 4)   # days 2,3 only
    assert abs(rate(trailing2, 3) - (10 + S * 0.5) / (20 + S)) < 1e-9, rate(trailing2, 3)  # days 1,2
    # the same trailing window applied to a LATER date from frozen tables
    prior, tables = ml.build_history_tables(base, lambda d: d, 2)
    assert abs(prior - 0.0) < 1e-12, prior       # last two fitting days (3, 4) are all zero


def test_apply_history_frozen_tables():
    fit = history_fixture(days=8)
    later = history_fixture(days=3, seed=11).withColumn("FlightDate", F.date_add(F.col("FlightDate"), 20)) \
        .withColumn("Reporting_Airline", F.when(F.col("flight_id").endswith("1"), "ZZ").otherwise(F.col("Reporting_Airline")))
    prior, tables = ml.build_history_tables(fit, lambda d: d)
    out = ml.apply_history(later, prior, tables)
    assert out.count() == later.count() and out.select("flight_id").distinct().count() == later.count()
    unseen = out.filter(F.col("Reporting_Airline") == "ZZ").select("hist_rate_airline").distinct().collect()
    assert len(unseen) == 1 and abs(unseen[0][0] - prior) < 1e-12, "unseen key must fall back to the prior"
    flipped = later.withColumn("label", 1.0 - F.col("label"))
    a = feature_map(out, ml.HISTORY_FEATURES)
    b = feature_map(ml.apply_history(flipped, prior, tables), ml.HISTORY_FEATURES)
    assert a == b, "validation/test labels must not influence their own history features"


def test_demand_counts_all_scheduled_flights_and_join_preserves_rows():
    d = date(2024, 10, 1)
    rows = [  # three ORD flights in hour 6 (one cancelled), one in hour 7, one at another airport
        (d, 0, 0, 1, 630, 900, 90.0, 400.0, "AA", 2, "ORD", "ATL", "0600-0659"),
        (d, 0, 0, 0, 659, 900, 90.0, 400.0, "DL", 2, "ORD", "DEN", "0600-0659"),
        (d, 1, 0, None, 601, 900, 90.0, 400.0, "UA", 2, "ORD", "LAX", "0600-0659"),
        (d, 0, 0, 0, 700, 900, 90.0, 400.0, "AA", 2, "ORD", "ATL", "0700-0759"),
        (d, 0, 0, 1, 630, 900, 90.0, 400.0, "AA", 2, "DEN", "ATL", "0600-0659"),
    ]
    clean = spark.createDataFrame(rows, CLEAN_SCHEMA)
    holidays = spark.createDataFrame([(d, "H")], "FlightDate date, HolidayPeriod string")
    demand = ml.compute_demand_table(clean)
    frame = ml.build_eligible_frame(clean, holidays, demand)
    assert frame.count() == 4, "cancelled flight must be excluded from rows"
    got = {(r["Origin"], r["ScheduledDepHour"]): r["OriginHourDemandLog"] for r in frame.collect()}
    import math
    assert abs(got[("ORD", 6.0)] - math.log1p(3)) < 1e-9, "demand must count the cancelled flight too"
    assert abs(got[("ORD", 7.0)] - math.log1p(1)) < 1e-9 and abs(got[("DEN", 6.0)] - math.log1p(1)) < 1e-9
    cut = ml.compute_demand_table(clean, cutoff_date=d)
    assert cut.count() == 0, "cutoff must exclude the cutoff date and later"
    try:
        ml.check_rows("probe", 4, 5)
        raise AssertionError("check_rows should reject a changed row count")
    except ml.PipelineError:
        pass


def test_threshold_search_uses_given_scores_only():
    rows = [(1.0, 0.9), (1.0, 0.8), (1.0, 0.3), (0.0, 0.7), (0.0, 0.2), (0.0, 0.1), (0.0, 0.05), (0.0, 0.04)]
    scored = spark.createDataFrame(rows, "label double, score double")
    thr, table = ml.threshold_search(scored, [0.1, 0.25, 0.5, 0.75, 0.85])
    by = {r[0]: r for r in table}
    assert by[0.25][1:4] == [3, 1, 0] and abs(by[0.25][6] - 0.857142857) < 1e-6, by[0.25]
    assert thr == 0.25, thr
    tp, fp, fn, tn = ml.confusion(scored, 0.5)
    assert (tp, fp, fn, tn) == (2, 1, 1, 4), (tp, fp, fn, tn)


def test_warmup_excludes_rows_but_keeps_their_labels_as_history():
    clean = synthetic_flights(days=20, per_day=40)
    holidays = spark.createDataFrame([(date(2024, 10, 10), "H")], "FlightDate date, HolidayPeriod string")
    eligible = ml.build_eligible_frame(clean, holidays, ml.compute_demand_table(clean))
    train = eligible.filter(F.col("FlightDate") < F.lit(date(2024, 10, 15)))
    val = eligible.filter(F.col("FlightDate") >= F.lit(date(2024, 10, 15)))
    keep, release = (lambda d: d), (lambda d: None)
    base_spec = dict(ml.EXPERIMENT_SPECS[0])
    warm_spec = dict(base_spec, warmup_days=7, name="warm")
    s0 = ml.prepare_stage("t0", base_spec, train, {"validation": val}, keep, release)
    s7 = ml.prepare_stage("t7", warm_spec, train, {"validation": val}, keep, release)
    expected = train.filter(F.col("FlightDate") >= F.lit(date(2024, 10, 8))).count()
    assert s7["fit_rows"] == expected and s0["fit_rows"] == train.count(), (s7["fit_rows"], expected, s0["fit_rows"])
    assert s7["applied"]["validation"].count() == val.count()


# --------------------------------------------------------------------------
# Integration tests (synthetic data, local mode; includes the output interface)
# --------------------------------------------------------------------------
def run_pipeline(root, name, mode, flip_from=None, extra=None):
    inputs = os.path.join(root, f"inputs_{name}")
    clean, holidays = write_inputs(inputs, flip_from)
    out = os.path.join(root, f"out_{name}")
    argv = [
    "--mode", mode,
    "--clean-path", clean,
    "--holidays-path", "file://" + os.path.abspath(holidays),
    "--local-results", os.path.join(out, "local"),
    # The Docker image sets fs.defaultFS=HDFS. Explicit file:// URIs keep these
    # synthetic-test artifacts local and make the test portable to that image.
    "--hdfs-results", "file://" + os.path.abspath(os.path.join(out, "hdfs_results")),
    "--hdfs-models", "file://" + os.path.abspath(os.path.join(out, "hdfs_models"))
] + (extra or [])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = ml.main(argv, spark=spark)
    return code, buffer.getvalue(), os.path.join(out, "local"), out


def tables_equal(a, b, tol=1e-9, ignore=("Seconds",)):
    """Equal shape/header; numeric cells within tol (parallel float summation order differs in the
    last digits between runs), every other cell exactly equal. Timing columns are ignored."""
    if len(a) != len(b) or a[0] != b[0]:
        return False, "different shape or header"
    skip = {a[0].index(c) for c in ignore if c in a[0]}
    for r, (ra, rb) in enumerate(zip(a, b)):
        for i, (x, y) in enumerate(zip(ra, rb)):
            if i in skip or x == y:
                continue
            try:
                if abs(float(x) - float(y)) <= tol:
                    continue
            except ValueError:
                pass
            return False, f"row {r} column {a[0][i]}: {x!r} != {y!r}"
    return True, ""


def numeric_column(table, name):
    header, body = table[0], table[1:]
    i = header.index(name)
    return [row[i] for row in body]


def test_default_namespace_is_safe_and_production_is_guarded():
    ctx = ml.Ctx(ml.parse_args([]))
    assert "experiments" in ctx.local_results and "experiments" in ctx.hdfs_results and "experiments" in ctx.hdfs_models
    assert ctx.local_results != ml.PRODUCTION_LOCAL_RESULTS
    for argv in (["--namespace", "production"], ["--namespace", "production", "--confirm-overwrite-baseline", "--mode", "select"]):
        try:
            ml.Ctx(ml.parse_args(argv))
            raise AssertionError(f"expected refusal for {argv}")
        except ml.PipelineError:
            pass
    prod = ml.Ctx(ml.parse_args(["--namespace", "production", "--confirm-overwrite-baseline"]))
    assert prod.local_results == "/results/ml" and prod.hdfs_results.endswith("/flights/results/ml")
    assert prod.hdfs_models.endswith("/flights/models")

    # A single overridden path must not bypass protection for the other production paths.
    unsafe_args = [
        ["--namespace", "production", "--confirm-overwrite-baseline", "--local-results", "/tmp/trial"],
        ["--namespace", "experiments", "--hdfs-results", ml.PRODUCTION_HDFS_RESULTS],
        ["--namespace", "experiments", "--hdfs-models", ml.PRODUCTION_HDFS_MODELS + "/candidate"],
        ["--namespace", "experiments", "--local-results", ml.PRODUCTION_LOCAL_RESULTS + "/trial"],
    ]
    for argv in unsafe_args:
        try:
            ml.Ctx(ml.parse_args(argv))
            raise AssertionError(f"expected refusal for a production-path bypass: {argv}")
        except ml.PipelineError:
            pass


def test_select_mode_never_reads_test_and_is_invariant_to_test_labels(root):
    flip_from = date(2024, 10, 1) + timedelta(days=28)   # test window starts at date index 28 of 40
    code, out, local, _ = run_pipeline(root, "sel_a", "select")
    assert code == 0 and "MLLIB_JOB_STATUS=SUCCEEDED" in out, out[-12000:]
    assert not os.path.exists(os.path.join(local, "metrics.csv")), "select mode must not write test metrics"
    assert "withheld" in out and "ML_TEST_ROWS" not in out
    cc = read_csv(os.path.join(local, "class_counts.csv"))
    assert cc[-1][0] == "test" and cc[-1][1] == "" and cc[-1][2] == "", cc[-1]
    for needed in ("experiment_comparison.csv", "validation_metrics.csv", "validation_threshold_search.csv",
                   "hyperparameter_search.csv", "frozen_config.json", "run_summary.txt", "eligibility_counts.csv"):
        assert os.path.exists(os.path.join(local, needed)), needed
    code_b, out_b, local_b, _ = run_pipeline(root, "sel_b", "select", flip_from=flip_from)
    assert code_b == 0, out_b[-2000:]
    for name in ("experiment_comparison.csv", "validation_threshold_search.csv", "validation_metrics.csv"):
        same, why = tables_equal(read_csv(os.path.join(local, name)), read_csv(os.path.join(local_b, name)))
        assert same, f"{name} changed when ONLY test-window labels changed: {why}"
    assert open(os.path.join(local, "frozen_config.json")).read() == open(os.path.join(local_b, "frozen_config.json")).read()
    table = read_csv(os.path.join(local, "experiment_comparison.csv"))
    experiments = set(numeric_column(table, "Experiment"))
    assert {"E0_reference", "E1_no_history", "E2_no_holiday_features", "E3_history_warmup7",
            "E4_history_trailing14", "E5_plus_scheduled_demand", "E8_extended_history", "E9_recent7_history",
            "E10_extra_demand", "E11_schedule_extras", "tune_LR", "tune_RF"} <= experiments, experiments


def test_all_mode_reconciles_and_freezes_before_test(root):
    flip_from = date(2024, 10, 1) + timedelta(days=28)
    subset = ["--experiments", "E0_reference,E1_no_history"]
    code, out, local, _ = run_pipeline(root, "all_a", "all", extra=subset)
    assert code == 0 and "MLLIB_JOB_STATUS=SUCCEEDED" in out, out[-2500:]
    assert out.index("DECISIONS_FROZEN=1") < out.index("Scoring held-out test window"), "test scored before freeze"
    metrics = read_csv(os.path.join(local, "metrics.csv"))
    assert metrics[0] == ["Model", "DecisionThreshold", "AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1",
                          "Recall_class1", "Accuracy"], metrics[0]
    assert [r[0] for r in metrics[1:]] == ["LogisticRegression", "RandomForest"]
    cc = read_csv(os.path.join(local, "class_counts.csv"))
    n = {r[0]: int(r[1]) for r in cc[1:]}
    elig = {r[0]: int(r[1]) for r in read_csv(os.path.join(local, "eligibility_counts.csv"))[1:]}
    assert sum(elig.values()) == 40 * 60, elig
    assert sum(n.values()) == elig["eligible"], (n, elig)
    conf = read_csv(os.path.join(local, "confusion_counts.csv"))
    for model in ("LogisticRegression", "RandomForest"):
        assert sum(int(r[3]) for r in conf[1:] if r[0] == model) == n["test"], model
    for needed in ("validation_threshold_search.csv", "random_forest_feature_importance_top30.csv", "run_summary.txt",
                   "baseline_metrics.csv", "ranking_metrics.csv", "metrics_by_holiday_slice.csv",
                   "threshold_transfer.csv", "frozen_config.json", "preprocessing_metadata.json"):
        assert os.path.exists(os.path.join(local, needed)), needed
    for model_dir in ("feature_pipeline", "logistic_regression", "random_forest"):
        assert os.path.isdir(os.path.join(root, "out_all_a", "hdfs_models", model_dir)), model_dir
    # decisions must not depend on test labels; metrics must (proves the test window is really scored)
    code_b, out_b, local_b, _ = run_pipeline(root, "all_b", "all", flip_from=flip_from, extra=subset)
    assert code_b == 0, out_b[-2500:]
    assert open(os.path.join(local, "frozen_config.json")).read() == open(os.path.join(local_b, "frozen_config.json")).read(), \
        "frozen decisions changed when only test labels changed"
    assert read_csv(os.path.join(local, "metrics.csv")) != read_csv(os.path.join(local_b, "metrics.csv")), \
        "test metrics did not respond to test labels: is the test window being scored?"
    frozen = json.load(open(os.path.join(local, "frozen_config.json")))
    assert frozen["frozen_before_test_scoring"] is True and set(frozen["thresholds_from_validation_F1"]) == {"LogisticRegression", "RandomForest"}


def test_failure_is_reported_not_swallowed(root):
    code, out, _, _ = run_pipeline(root, "bad", "select", extra=["--experiments", "NOT_AN_EXPERIMENT"])
    assert code == 2 and "MLLIB_JOB_STATUS=FAILED" in out and "MLLIB_JOB_STATUS=SUCCEEDED" not in out, out[-800:]



# --------------------------------------------------------------------------
# Tests for the added feature engineering (extended / recent history, extra demand, schedule extras)
# --------------------------------------------------------------------------
def history_fixture_ext():
    """history_fixture plus the derived key columns the extended history groups need."""
    base = history_fixture(days=14, per_day=40, seed=5)
    return (base.withColumn("DayOfWeekCat", F.concat(F.lit("DOW_"), F.dayofweek("FlightDate").cast("string")))
            .withColumn("ScheduledArrHour", (F.abs(F.hash("flight_id")) % F.lit(4)).cast("double"))
            .withColumn("ScheduledDepHour", (F.abs(F.hash("flight_id", F.lit("x"))) % F.lit(5)).cast("double")))


def apply_all_groups(spec):
    """Exactly what prepare_stage does to the fitting rows: chain one past-only pass per history group."""
    def fn(df):
        out = df
        for group in ml.history_groups(spec):
            out = ml.apply_history_training_past_only(out, group["window_days"], group["keys_map"])
        return out
    return fn


def test_added_history_groups_are_past_only_and_row_preserving():
    base = history_fixture_ext()
    n_in = base.count()
    dates = [date(2024, 10, 1) + timedelta(days=i) for i in (0, 2, 5, 8, 11)]
    for name in ("E8_extended_history", "E9_recent7_history"):
        spec = next(x for x in ml.EXPERIMENT_SPECS if x["name"] == name)
        groups = ml.history_groups(spec)
        cols = [c for g in groups for c in g["keys_map"]]
        assert len(cols) == len(set(cols)), f"{name}: duplicate history feature names {cols}"
        fn = apply_all_groups(spec)
        out = fn(base)
        assert out.count() == n_in, f"{name}: history joins changed the row count"
        assert out.select("flight_id").distinct().count() == n_in, f"{name}: duplicated flight ids"
        assert set(cols) <= set(out.columns), name
        bad = future_dependence_violations(fn, cols, base, dates)
        assert not bad, f"{name}: labels on/after D changed history features on/before D: {bad[:3]}"
    ext = ml.history_groups(next(x for x in ml.EXPERIMENT_SPECS if x["name"] == "E8_extended_history"))
    rec = ml.history_groups(next(x for x in ml.EXPERIMENT_SPECS if x["name"] == "E9_recent7_history"))
    assert len(ext) == 1 and len(ext[0]["keys_map"]) == len(ml.HISTORY_KEYS) + len(ml.EXTRA_HISTORY_KEYS)
    assert [g["name"] for g in rec] == ["base", "recent"] and rec[1]["window_days"] == 7


def test_recent_group_uses_only_last_n_earlier_dates():
    rows = []
    for d, lab in {1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}.items():
        for k in range(10):
            rows.append((f"r{d}_{k}", date(2024, 10, d), lab, "AA", "AAA", "BBB", "0600-0659"))
    base = spark.createDataFrame(rows, "flight_id string, FlightDate date, label double, "
                                 "Reporting_Airline string, Origin string, Dest string, DepTimeBlk string")
    S = ml.HISTORY_SMOOTHING
    out = ml.apply_history_training_past_only(base, 2, ml.RECENT_HISTORY_KEYS)
    got = out.filter(F.col("FlightDate") == F.lit(date(2024, 10, 5))).select("recent_rate_airline").first()[0]
    # window = days 3 and 4 only (all zeros); prior from the same two days = 0.0
    assert abs(got - (0 + S * 0.0) / (20 + S)) < 1e-9, got
    got4 = out.filter(F.col("FlightDate") == F.lit(date(2024, 10, 4))).select("recent_rate_airline").first()[0]
    # window = days 2 and 3: 10 positives of 20; prior 0.5
    assert abs(got4 - (10 + S * 0.5) / (20 + S)) < 1e-9, got4


def test_added_demand_and_schedule_features_exact():
    import math
    d = date(2024, 10, 1)
    rows = [  # all arrive 09:00 (CRSArrTime 900); ORD flights in dep hours 6 (x3, one cancelled) and 7 (x1)
        (d, 0, 0, 1, 630, 900, 90.0, 400.0, "AA", 2, "ORD", "ATL", "0600-0659"),
        (d, 0, 0, 0, 659, 900, 90.0, 400.0, "DL", 2, "ORD", "DEN", "0600-0659"),
        (d, 1, 0, None, 601, 900, 90.0, 400.0, "UA", 2, "ORD", "LAX", "0600-0659"),
        (d, 0, 0, 0, 700, 900, 90.0, 400.0, "AA", 2, "ORD", "ATL", "0700-0759"),
        (d, 0, 0, 1, 630, 900, 0.0, 400.0, "AA", 2, "DEN", "ATL", "0600-0659"),
    ]
    clean = spark.createDataFrame(rows, CLEAN_SCHEMA)
    holidays = spark.createDataFrame([(d, "H")], "FlightDate date, HolidayPeriod string")
    frame = ml.build_eligible_frame(clean, holidays, ml.compute_demand_table(clean))
    assert frame.count() == 4, "row count changed by the extra demand joins (or cancelled row leaked in)"
    by = {(r["Origin"], r["Dest"], r["ScheduledDepHour"]): r for r in frame.collect()}
    f1, f4 = by[("ORD", "ATL", 6.0)], by[("ORD", "ATL", 7.0)]
    assert abs(f1["OriginDayDemandLog"] - math.log1p(4)) < 1e-9, "origin-day demand must include the cancelled flight"
    assert abs(f1["DestHourDemandLog"] - math.log1p(3)) < 1e-9, "dest-hour demand counts ATL arrivals at 09:00"
    assert abs(f1["OriginPrevHourDemandLog"] - 0.0) < 1e-9, "no flights in the hour before 06:00"
    assert abs(f4["OriginPrevHourDemandLog"] - math.log1p(3)) < 1e-9, "previous hour (06:00) had 3 scheduled flights"
    assert abs(f1["SchedSpeedMph"] - 400.0 / 1.5) < 1e-9 and abs(f1["DistanceLog"] - math.log1p(400.0)) < 1e-9
    zero_block = by[("DEN", "ATL", 6.0)]
    assert zero_block["SchedSpeedMph"] is None, "non-positive block time must give a null speed, not inf"
    # the imputer must cope with that null: build the vectors for the spec that uses it
    spec = dict(next(x for x in ml.EXPERIMENT_SPECS if x["name"] == "E11_schedule_extras"), use_history=False)
    model = ml.make_pipeline(spec).fit(frame)           # history off: this frame has no history columns
    transformed = model.transform(frame)
    assert transformed.count() == 4
    imputed = {r["Origin"] + r["Dest"] + str(r["ScheduledDepHour"]): r["SchedSpeedMph__imputed"]
               for r in transformed.collect()}
    assert imputed["DENATL6.0"] is not None and abs(imputed["DENATL6.0"] - 400.0 / 1.5) < 1e-9, \
        "null speed must be imputed with the fitting-window median, not left null or infinite"


def test_combine_specs_merges_single_factor_changes():
    ref = ml.EXPERIMENT_SPECS[0]
    wins = [x for x in ml.EXPERIMENT_SPECS if x["name"] in ("E8_extended_history", "E11_schedule_extras")]
    merged = ml.combine_specs(ref, wins)
    assert merged["history_set"] == "extended" and merged["sched_extras"] is True
    assert merged["use_history"] is True and merged["extra_demand"] is False
    no_hist = ml.combine_specs(ref, [x for x in ml.EXPERIMENT_SPECS if x["name"] in ("E1_no_history", "E8_extended_history")])
    assert no_hist["use_history"] is False and no_hist["history_set"] == "base" and no_hist["recent_days"] is None
    impute, plain, hist, cat = ml.spec_columns(merged)
    assert "SchedSpeedMph" in impute and len(hist) == len(ml.HISTORY_KEYS) + len(ml.EXTRA_HISTORY_KEYS)


def test_unknown_extra_model_is_rejected():
    try:
        ml.Ctx(ml.parse_args(["--extra-models", "gbt,catboost"]))
        raise AssertionError("expected a clear refusal for an unsupported extra model")
    except ml.PipelineError as error:
        assert "catboost" in str(error)
    ctx = ml.Ctx(ml.parse_args(["--extra-models", "MLP, gbt"]))
    assert ctx.extra_models == ["gbt", "mlp"], ctx.extra_models


def test_extra_models_full_pipeline_and_decision_invariance(root):
    flip_from = date(2024, 10, 1) + timedelta(days=28)
    args = ["--experiments", "E0_reference", "--extra-models", "gbt,mlp"]
    code, out, local, out_dir = run_pipeline(root, "xm_a", "all", extra=args)
    assert code == 0 and "MLLIB_JOB_STATUS=SUCCEEDED" in out, out[-2500:]
    assert out.index("DECISIONS_FROZEN=1") < out.index("Scoring held-out test window"), "test scored before freeze"
    metrics = read_csv(os.path.join(local, "metrics.csv"))
    assert [r[0] for r in metrics[1:]] == ["LogisticRegression", "RandomForest", "GBTClassifier",
                                           "MultilayerPerceptron"], [r[0] for r in metrics[1:]]
    n_test = {r[0]: int(r[1]) for r in read_csv(os.path.join(local, "class_counts.csv"))[1:]}["test"]
    conf = read_csv(os.path.join(local, "confusion_counts.csv"))
    for model in ("GBTClassifier", "MultilayerPerceptron"):
        assert sum(int(r[3]) for r in conf[1:] if r[0] == model) == n_test, model
    search = read_csv(os.path.join(local, "hyperparameter_search.csv"))
    for family in ("GBTClassifier", "MultilayerPerceptron"):
        chosen = [r for r in search[1:] if r[0] == family and r[4] == "yes"]
        assert len(chosen) == 1 and len([r for r in search[1:] if r[0] == family]) == 2, (family, chosen)
    thr_models = {r[0] for r in read_csv(os.path.join(local, "validation_threshold_search.csv"))[1:]}
    assert {"GBTClassifier", "MultilayerPerceptron"} <= thr_models, thr_models
    frozen = json.load(open(os.path.join(local, "frozen_config.json")))
    assert set(frozen["extra_models"]) == {"GBTClassifier", "MultilayerPerceptron"}, frozen["extra_models"]
    assert set(frozen["thresholds_from_validation_F1"]) == {"LogisticRegression", "RandomForest",
                                                            "GBTClassifier", "MultilayerPerceptron"}
    for directory in ("gbt", "mlp", "logistic_regression", "random_forest", "feature_pipeline"):
        assert os.path.isdir(os.path.join(out_dir, "hdfs_models", directory)), directory
    comparison = read_csv(os.path.join(local, "experiment_comparison.csv"))
    assert any(r[comparison[0].index("Experiment")] == "tune_GBTClassifier" for r in comparison[1:])
    # decisions (including the extras' hyperparameters and thresholds) must not depend on test labels
    code_b, out_b, local_b, _ = run_pipeline(root, "xm_b", "all", flip_from=flip_from, extra=args)
    assert code_b == 0, out_b[-2500:]
    assert open(os.path.join(local, "frozen_config.json")).read() == open(os.path.join(local_b, "frozen_config.json")).read(), \
        "frozen decisions (incl. extra models) changed when only test labels changed"
    assert read_csv(os.path.join(local, "metrics.csv")) != read_csv(os.path.join(local_b, "metrics.csv"))


def test_unsupported_extra_model_fails_loudly(root):
    code, out, _, _ = run_pipeline(root, "xm_bad", "select", extra=["--extra-models", "catboost"])
    assert code == 2 and "MLLIB_JOB_STATUS=FAILED" in out and "catboost" in out, out[-600:]


def main():
    global spark, ml
    ml = load_ml_module(SCRIPT_PATH)
    builder = (SparkSession.builder.appName("BDA-ML-Pipeline-Synthetic-Tests").master("local[2]")
               .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4")
               .config("spark.default.parallelism", "2"))
    # Same session settings the production script applies when it creates its own session.
    if os.environ.get("ML_TEST_USE_SCRIPT_SESSION_CONF", "1") == "1":
        for key, value in ml.SPARK_SESSION_CONF.items():
            builder = builder.config(key, value)
    # Diagnostics only: ML_TEST_SPARK_CONF="k=v,k=v"
    for pair in filter(None, os.environ.get("ML_TEST_SPARK_CONF", "").split(",")):
        key, _, value = pair.partition("=")
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    root = tempfile.mkdtemp(prefix="ml_pipeline_tests_")
    unit = [test_minute_of_day, test_windows_match_baseline_split, test_validate_schema,
            test_eligibility_categories_are_exclusive_and_sum, test_leakage_guard_and_provenance,
            test_history_training_past_only_property, test_property_check_detects_the_old_leaky_design,
            test_history_key_isolation_and_first_day_fallback, test_trailing_window_exact_values,
            test_apply_history_frozen_tables, test_demand_counts_all_scheduled_flights_and_join_preserves_rows,
            test_threshold_search_uses_given_scores_only, test_warmup_excludes_rows_but_keeps_their_labels_as_history,
            test_default_namespace_is_safe_and_production_is_guarded,
            test_added_history_groups_are_past_only_and_row_preserving, test_recent_group_uses_only_last_n_earlier_dates,
            test_added_demand_and_schedule_features_exact, test_combine_specs_merges_single_factor_changes,
            test_unknown_extra_model_is_rejected]
    integration = [test_select_mode_never_reads_test_and_is_invariant_to_test_labels,
                   test_all_mode_reconciles_and_freezes_before_test, test_failure_is_reported_not_swallowed,
                   test_extra_models_full_pipeline_and_decision_invariance, test_unsupported_extra_model_fails_loudly]
    failures = 0
    wanted = sys.argv[1:]          # optional: run only tests whose name contains one of these words
    try:
        for test in unit + integration:
            if wanted and not any(w in test.__name__ for w in wanted):
                continue
            started = time.time()
            try:
                test(root) if test in integration else test()
                print(f"PASS {test.__name__} ({time.time() - started:.1f}s)", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL {test.__name__}", flush=True)
                traceback.print_exc()
    finally:
        spark.stop()
        shutil.rmtree(root, ignore_errors=True)
    if failures:
        print(f"ML_PIPELINE_TESTS_STATUS=FAILED ({failures} failing)", flush=True)
        return 1
    print("ML_PIPELINE_TESTS_STATUS=PASSED", flush=True)
    print("SYNTHETIC/LOCAL tests only: no real BTS data, no YARN, no model-quality claim.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
