#!/usr/bin/env python3
"""Spark MLlib classification of BTS flight arrival delays (ArrDel15).

Question: can we predict, using only information available BEFORE departure,
whether an eligible scheduled domestic flight arrives >= 15 minutes late?
This is an observational prediction task on three months of data. It is NOT a
causal estimate of any holiday effect (see the descriptive Spark SQL job for that).

Design (see the run summary for the numbers of an actual run)
-------------------------------------------------------------
1. Eligibility: not cancelled, not diverted, ArrDel15 in {0, 1}. The number of
   flights removed for each reason is printed and saved.
2. Leakage control: an explicit allow-list of pre-departure source columns, a
   hard runtime check against a deny-list of post-departure/outcome columns,
   and no raw FlightDate / Month / DayOfYear / Year predictors (see 5).
3. Split by whole FlightDate, chronologically (identical windows to the earlier run):
   train -> validation -> final test. The test window is never used to choose
   features, hyperparameters or thresholds.
4. Preprocessing is fitted on the fitting window only (imputer medians, category
   indexers, historical-rate tables, prior).
5. Features (all known before departure):
   * scheduled departure/arrival time parsed from HHMM into minute-of-day, then
     cyclic sin/cos (23:59 and 00:01 are neighbours) plus the scheduled hour;
   * scheduled elapsed time, distance;
   * carrier, day of week (one-hot), holiday period and a holiday flag;
   * smoothed HISTORICAL delay rates for carrier, origin, destination, route and
     origin x scheduled-departure-block. For each fitting row, these rates and the
     global prior use only strictly earlier flight dates. Rows in later windows use
     rates fitted from the completed fitting window. This prevents future-date label
     leakage and avoids one-hot encoding hundreds of airports.
   Raw Year/Month/DayofMonth/DayOfYear are deliberately NOT used: in a chronological
   split the test dates lie outside the training date range, so these columns can
   only memorise training days (trees cannot extrapolate) or extrapolate arbitrarily.
6. Models: Logistic Regression (baseline) and Random Forest, each tuned over a
   SMALL grid by validation AUC-PR (threshold-free, suited to imbalance).
   The class-1 probability threshold is then chosen by validation F1 only.
7. Final models are refitted on train + validation (all data before the test
   window, using the already-selected hyperparameters and threshold) and scored
   once on the held-out test window. A refit gives the model the most recent data
   (the delay rate drifts upward over time) and exposure to Thanksgiving week.
8. Reported: AUC-ROC, AUC-PR, class-1 precision/recall/F1, accuracy, confusion
   counts, prevalence, trivial baselines, ranking lift, per-holiday-slice metrics
   and a validation-only ablation (with vs without historical-rate features).

Limitations that cannot be fixed by code: no weather, no inbound-aircraft status
and no same-day network state are allowed (they are not known before departure
or would leak), so signal is modest; delay prevalence drifts over the three
months; Thanksgiving/Christmas categories are rare or absent in training windows.
Better metrics are not guaranteed; whatever the held-out test window gives is
what must be reported.
"""

import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window, functions as F, types as T

# --------------------------------------------------------------------------
# Paths (unchanged from the earlier run)
# --------------------------------------------------------------------------
CLEAN_PATH = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS_PATH = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
HDFS_RESULTS = "hdfs://master:9000/user/root/flights/results/ml"
HDFS_MODELS = "hdfs://master:9000/user/root/flights/models"
LOCAL_RESULTS = "/results/ml"

# --------------------------------------------------------------------------
# Settings (edit here to make the job lighter or heavier)
# --------------------------------------------------------------------------
VALIDATION_START_DATE_FRACTION = 0.60   # same windows as the earlier run
TEST_START_DATE_FRACTION = 0.70
THRESHOLD_CANDIDATES = [round(0.03 + 0.01 * i, 2) for i in range(58)]   # 0.03 ... 0.60
TOP_FRACTIONS = [0.10, 0.20, 0.30]      # "flag the riskiest X% of flights" lift table
SEED = 42

HISTORY_SMOOTHING = 100.0               # pseudo-flights pulling rates toward the prior
HISTORY_KEYS = {
    "hist_rate_airline": ["Reporting_Airline"],
    "hist_rate_origin": ["Origin"],
    "hist_rate_dest": ["Dest"],
    "hist_rate_route": ["Origin", "Dest"],
    "hist_rate_origin_depblk": ["Origin", "DepTimeBlk"],
}

LR_MAX_ITER = 30
LR_REG_GRID = [0.001, 0.01, 0.1]        # L2 strength, 3 candidates

RF_NUM_TREES = 30
RF_CONFIGS = [                          # 2 candidates, chosen on validation AUC-PR
    {"maxDepth": 5, "minInstancesPerNode": 100},
    {"maxDepth": 7, "minInstancesPerNode": 100},
]
RF_MAX_BINS = 32
RF_SUBSAMPLING_RATE = 0.5
RF_FEATURE_SUBSET = "onethird"
RF_MAX_MEMORY_MB = 128                  # small histogram budget for 512 MB executors

# --------------------------------------------------------------------------
# Feature definitions and leakage rules
# --------------------------------------------------------------------------
# Source columns the predictors are built from (all known before departure).
# Origin / Dest / DepTimeBlk are used only as keys of the historical-rate tables.
FEATURE_SOURCE_COLUMNS = [
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "Reporting_Airline",
    "DayOfWeek", "Origin", "Dest", "DepTimeBlk", "HolidayPeriod",
]
REQUIRED_CLEAN_COLUMNS = ["FlightDate", "Cancelled", "Diverted", "ArrDel15"] + [
    c for c in FEATURE_SOURCE_COLUMNS if c != "HolidayPeriod"
]
FORBIDDEN_SOURCE_COLUMNS = {
    "ArrDel15", "ArrDelay", "ArrDelayMinutes", "ArrivalDelayGroups",
    "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups", "DepTime",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "ArrTime", "ActualElapsedTime",
    "AirTime", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "Cancelled", "CancellationCode", "Diverted", "Tail_Number",
}

PLAIN_NUMERIC_FEATURES = [
    "DepMinuteSin", "DepMinuteCos", "ArrMinuteSin", "ArrMinuteCos",
    "ScheduledDepHour", "ScheduledArrHour", "IsHolidayWindow",
]
IMPUTE_FEATURES = ["CRSElapsedTime", "Distance"]
HISTORY_FEATURES = list(HISTORY_KEYS.keys())
CATEGORICAL_FEATURES = ["Reporting_Airline", "DayOfWeekCat", "HolidayPeriod"]

T0 = time.time()


class PipelineError(Exception):
    """A clear, expected failure (bad input, empty split, ...)."""


def log(message):
    print(message, flush=True)


def stamp(message):
    log(f"[{time.time() - T0:8.1f}s] {message}")


def assert_no_leakage():
    bad = sorted(
        c for c in FEATURE_SOURCE_COLUMNS
        if c in FORBIDDEN_SOURCE_COLUMNS or c.startswith("Div")
    )
    if bad:
        raise PipelineError(f"Leakage guard: forbidden source columns in feature list: {bad}")


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def preflight_local_output():
    os.makedirs(LOCAL_RESULTS, exist_ok=True)
    probe = os.path.join(LOCAL_RESULTS, ".write_probe")
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write("ok")
    os.remove(probe)


def save_table(spark, filename, header, rows):
    """Write a small result table to the host mount and to HDFS (as one CSV part file)."""
    local_path = os.path.join(LOCAL_RESULTS, filename)
    with open(local_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    string_rows = [tuple("" if v is None else str(v) for v in row) for row in rows]
    schema = T.StructType([T.StructField(name, T.StringType(), True) for name in header])
    hdfs_path = f"{HDFS_RESULTS}/{filename[:-4] if filename.endswith('.csv') else filename}"
    (spark.createDataFrame(string_rows, schema)
     .coalesce(1).write.mode("overwrite").option("header", "true").csv(hdfs_path))
    log(f"Saved {filename}: local={local_path} hdfs={hdfs_path}")


def save_text_hdfs(spark, relative_name, text):
    path = f"{HDFS_RESULTS}/{relative_name}"
    spark.createDataFrame([(text,)], ["value"]).coalesce(1).write.mode("overwrite").text(path)
    return path


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------
def minute_of_day(column):
    """Parse an HHMM integer (e.g. 1745, 2400) into minutes after midnight, else null."""
    hh = F.floor(column / F.lit(100))
    mm = column - hh * F.lit(100)
    valid = column.isNotNull() & (column >= 0) & (column <= 2400) & (mm >= 0) & (mm < 60)
    return F.when(valid, F.pmod(hh * F.lit(60) + mm, F.lit(1440))).cast("double")


def build_history_tables(fit_base, keep):
    """Per-key (positives, flights) counts and the prior, from the fitting window only."""
    prior = float(fit_base.agg(F.avg("label")).first()[0])
    tables = {}
    for name, keys in HISTORY_KEYS.items():
        table = keep(
            fit_base.groupBy(*keys).agg(
                F.sum("label").alias("hist_pos"), F.count(F.lit(1)).alias("hist_n")
            )
        )
        rows = table.count()
        log(f"  historical-rate table {name} keys={keys}: {rows:,} groups")
        tables[name] = table
    return prior, tables


def apply_history(df, prior, tables):
    """Apply smoothed rates from a completed, strictly earlier fitting window.

    This function is for validation/test scoring data only. Since the fitting window
    ends before those rows occur, all labels used in these aggregate tables are
    historically available at scoring time.
    """
    out = df
    pseudo_positives = F.lit(float(HISTORY_SMOOTHING * prior))
    pseudo_flights = F.lit(float(HISTORY_SMOOTHING))
    for name, keys in HISTORY_KEYS.items():
        table = (tables[name]
                 .withColumnRenamed("hist_pos", f"{name}_pos")
                 .withColumnRenamed("hist_n", f"{name}_n"))
        out = out.join(table, on=keys, how="left")
        positives = F.coalesce(F.col(f"{name}_pos"), F.lit(0.0))
        flights = F.coalesce(F.col(f"{name}_n"), F.lit(0)).cast("double")
        out = out.withColumn(name, (positives + pseudo_positives) / (flights + pseudo_flights))
        out = out.drop(f"{name}_pos", f"{name}_n")
    return out


def apply_history_training_past_only(fit_base):
    """Create fitting-row history features using strictly earlier flight dates.

    For every fitting row on date D, each category rate uses observations from
    dates strictly before D. Unlike a leave-one-day-out aggregate across the whole
    fitting window, this cannot use labels from later dates to construct an earlier
    row's predictors. The global prior is likewise based only on earlier dates;
    the first date uses a neutral 0.5 prior because no earlier observations exist.
    """
    global_daily = fit_base.groupBy("FlightDate").agg(
        F.sum("label").alias("_global_day_pos"),
        F.count(F.lit(1)).alias("_global_day_n"),
    )
    global_window = Window.orderBy("FlightDate").rowsBetween(Window.unboundedPreceding, -1)
    global_daily = (
        global_daily
        .withColumn("_global_prev_pos", F.sum("_global_day_pos").over(global_window))
        .withColumn("_global_prev_n", F.sum("_global_day_n").over(global_window))
        .withColumn(
            "_global_prior",
            F.when(
                F.coalesce(F.col("_global_prev_n"), F.lit(0)) > 0,
                F.col("_global_prev_pos") / F.col("_global_prev_n"),
            ).otherwise(F.lit(0.5)),
        )
        .select("FlightDate", "_global_prior")
    )

    out = fit_base
    for name, keys in HISTORY_KEYS.items():
        daily_window = Window.partitionBy(*keys).orderBy("FlightDate").rowsBetween(
            Window.unboundedPreceding, -1
        )
        daily_rates = (
            fit_base.groupBy(*(keys + ["FlightDate"]))
            .agg(
                F.sum("label").alias("_day_pos"),
                F.count(F.lit(1)).alias("_day_n"),
            )
            .withColumn("_prev_pos", F.sum("_day_pos").over(daily_window))
            .withColumn("_prev_n", F.sum("_day_n").over(daily_window))
            .join(F.broadcast(global_daily), on="FlightDate", how="left")
        )
        prev_pos = F.coalesce(F.col("_prev_pos"), F.lit(0.0))
        prev_n = F.coalesce(F.col("_prev_n"), F.lit(0)).cast("double")
        prior = F.coalesce(F.col("_global_prior"), F.lit(0.5))
        rate = (prev_pos + F.lit(HISTORY_SMOOTHING) * prior) / (
            prev_n + F.lit(HISTORY_SMOOTHING)
        )
        daily_rates = daily_rates.select(*(keys + ["FlightDate"]), rate.alias(name))
        out = out.join(daily_rates, on=keys + ["FlightDate"], how="left")
    return out


def make_pipeline():
    """Preprocessing fitted on the fitting window only (imputer, indexers, one-hot, assembler)."""
    imputed = [f"{c}__imputed" for c in IMPUTE_FEATURES]
    index_cols = [f"{c}__idx" for c in CATEGORICAL_FEATURES]
    ohe_cols = [f"{c}__ohe" for c in CATEGORICAL_FEATURES]
    stages = [Imputer(strategy="median", inputCols=IMPUTE_FEATURES, outputCols=imputed)]
    stages.extend(
        StringIndexer(inputCol=c, outputCol=i, handleInvalid="keep")
        for c, i in zip(CATEGORICAL_FEATURES, index_cols)
    )
    stages.append(OneHotEncoder(inputCols=index_cols, outputCols=ohe_cols, handleInvalid="keep", dropLast=False))
    main_inputs = imputed + PLAIN_NUMERIC_FEATURES + HISTORY_FEATURES + ohe_cols
    nohist_inputs = imputed + PLAIN_NUMERIC_FEATURES + ohe_cols
    stages.append(VectorAssembler(inputCols=main_inputs, outputCol="features", handleInvalid="keep"))
    nohist_assembler = VectorAssembler(inputCols=nohist_inputs, outputCol="features_nohist", handleInvalid="keep")
    return Pipeline(stages=stages), nohist_assembler


def feature_names(vec_df):
    """Recover VectorAssembler feature names from column metadata."""
    metadata = vec_df.schema["features"].metadata.get("ml_attr", {})
    names = [f"feature_{i}" for i in range(int(metadata.get("num_attrs", 0)))]
    for attr_list in metadata.get("attrs", {}).values():
        for item in attr_list:
            idx = int(item.get("idx", -1))
            if 0 <= idx < len(names):
                names[idx] = item.get("name", names[idx])
    return names


def prepare_stage(label, fit_base, apply_bases, with_ablation, keep, release):
    """Build historical-rate features and feature vectors for one fitting window."""
    stamp(f"{label}: historical-rate tables (fitting window only)")
    prior, tables = build_history_tables(fit_base, keep)
    stamp(f"{label}: prior delay rate in fitting window = {prior:.4f}")

    fit_h = keep(apply_history_training_past_only(fit_base))
    fit_rows = fit_h.count()
    applied_h = {}
    for name, df in apply_bases.items():
        applied_h[name] = keep(apply_history(df, prior, tables))
        applied_h[name].count()

    stamp(f"{label}: fitting preprocessing pipeline on the fitting window only")
    pipeline, nohist_assembler = make_pipeline()
    pipeline_model = pipeline.fit(fit_h)

    def vectorize(df):
        out = pipeline_model.transform(df)
        columns = ["label", "HolidayPeriod", "features"]
        if with_ablation:
            out = nohist_assembler.transform(out)
            columns.append("features_nohist")
        return out.select(*columns)

    fit_vec = keep(vectorize(fit_h))
    fit_vec.count()
    applied_vec = {}
    for name, df in applied_h.items():
        applied_vec[name] = keep(vectorize(df))
        applied_vec[name].count()
    release(fit_h)
    for df in applied_h.values():
        release(df)
    names = feature_names(fit_vec)
    stamp(f"{label}: feature vectors ready ({fit_rows:,} fitting rows, {len(names)} features)")
    return {
        "prior": prior, "tables": tables, "pipeline_model": pipeline_model,
        "fit_vec": fit_vec, "applied": applied_vec, "feature_names": names,
    }


# --------------------------------------------------------------------------
# Models, scoring and metrics
# --------------------------------------------------------------------------
def fit_logistic(vec_df, reg_param, features_col="features"):
    return LogisticRegression(
        labelCol="label", featuresCol=features_col, maxIter=LR_MAX_ITER,
        regParam=float(reg_param), elasticNetParam=0.0, standardization=True,
    ).fit(vec_df)


def fit_forest(vec_df, config):
    return RandomForestClassifier(
        labelCol="label", featuresCol="features", numTrees=RF_NUM_TREES,
        maxDepth=int(config["maxDepth"]), minInstancesPerNode=int(config["minInstancesPerNode"]),
        maxBins=RF_MAX_BINS, subsamplingRate=RF_SUBSAMPLING_RATE,
        featureSubsetStrategy=RF_FEATURE_SUBSET, maxMemoryInMB=RF_MAX_MEMORY_MB, seed=SEED,
    ).fit(vec_df)


def score_dataframe(model, vec_df, keep):
    """Class-1 probability per flight; cached because several metrics reuse it."""
    scored = keep(
        model.transform(vec_df).select(
            "label", "HolidayPeriod", "probability",
            vector_to_array(F.col("probability"))[1].alias("score"),
        )
    )
    scored.count()
    return scored


def auc_pair(scored):
    roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC").evaluate(scored)
    pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR").evaluate(scored)
    return float(roc), float(pr)


def prf(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return precision, recall, f1, accuracy


def confusion(scored, threshold):
    predicted = F.col("score") >= F.lit(float(threshold))
    row = scored.agg(
        F.sum(F.when(predicted & (F.col("label") == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when(predicted & (F.col("label") == 0.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((~predicted) & (F.col("label") == 1.0), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((~predicted) & (F.col("label") == 0.0), 1).otherwise(0)).alias("tn"),
    ).first()
    return tuple(int(row[k] or 0) for k in ("tp", "fp", "fn", "tn"))


def threshold_search(scored_validation, candidates):
    """Validation-only threshold grid. Returns (best_threshold, rows)."""
    rows = []
    for start in range(0, len(candidates), 12):
        chunk = candidates[start:start + 12]
        expressions = []
        for i, threshold in enumerate(chunk):
            predicted = F.col("score") >= F.lit(float(threshold))
            expressions.append(F.sum(F.when(predicted & (F.col("label") == 1.0), 1).otherwise(0)).alias(f"tp_{i}"))
            expressions.append(F.sum(F.when(predicted & (F.col("label") == 0.0), 1).otherwise(0)).alias(f"fp_{i}"))
            expressions.append(F.sum(F.when((~predicted) & (F.col("label") == 1.0), 1).otherwise(0)).alias(f"fn_{i}"))
        totals = scored_validation.agg(*expressions).first()
        for i, threshold in enumerate(chunk):
            tp, fp, fn = (int(totals[f"{k}_{i}"] or 0) for k in ("tp", "fp", "fn"))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append([float(threshold), tp, fp, fn, precision, recall, f1])
    best = max(rows, key=lambda r: (r[6], r[4], r[0]))   # F1, then precision, then higher threshold
    return best[0], rows


def ranking_metrics(scored, fractions, total, positives):
    """Precision/recall/lift when flagging the riskiest X% of flights (labels not used to rank)."""
    cutoffs = scored.approxQuantile("score", [1.0 - f for f in fractions], 0.001)
    expressions = []
    for i, cutoff in enumerate(cutoffs):
        flagged = F.col("score") >= F.lit(float(cutoff))
        expressions.append(F.sum(F.when(flagged, 1).otherwise(0)).alias(f"n_{i}"))
        expressions.append(F.sum(F.when(flagged & (F.col("label") == 1.0), 1).otherwise(0)).alias(f"p_{i}"))
    totals = scored.agg(*expressions).first()
    prevalence = positives / total if total else 0.0
    rows = []
    for i, fraction in enumerate(fractions):
        flagged_n = int(totals[f"n_{i}"] or 0)
        flagged_p = int(totals[f"p_{i}"] or 0)
        precision = flagged_p / flagged_n if flagged_n else 0.0
        recall = flagged_p / positives if positives else 0.0
        lift = precision / prevalence if prevalence else 0.0
        rows.append([fraction, flagged_n / total if total else 0.0, flagged_n, precision, recall, lift])
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    spark = SparkSession.builder.appName("BDA-03-Flight-Delay-MLlib").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    cached = []

    def keep(df):
        df.persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(df)
        return df

    def release(df):
        try:
            df.unpersist()
        except Exception:
            pass
        cached[:] = [d for d in cached if d is not df]

    try:
        assert_no_leakage()
        preflight_local_output()
        log(f"Spark version: {spark.version}")
        log(f"Predictor source columns (allow-list): {FEATURE_SOURCE_COLUMNS}")

        # ---------------- inputs ----------------
        stamp(f"Reading cleaned Parquet: {CLEAN_PATH}")
        flights = spark.read.parquet(CLEAN_PATH)
        missing = [c for c in REQUIRED_CLEAN_COLUMNS if c not in flights.columns]
        if missing:
            raise PipelineError(f"Cleaned Parquet is missing required columns: {missing}")

        stamp(f"Reading holiday calendar: {HOLIDAYS_PATH}")
        holidays = (
            spark.read.option("header", "true").csv(HOLIDAYS_PATH)
            .select("FlightDate", "HolidayPeriod")
            .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
            .withColumn("HolidayPeriod", F.trim(F.col("HolidayPeriod")))
            .filter(F.col("FlightDate").isNotNull() & F.col("HolidayPeriod").isNotNull())
        )
        if holidays.groupBy("FlightDate").count().filter(F.col("count") > 1).limit(1).count():
            raise PipelineError("holidays.csv contains duplicate FlightDate values; join would duplicate flights.")

        # ---------------- eligibility accounting ----------------
        category = (
            F.when(F.col("Cancelled").isNull() | F.col("Diverted").isNull(), "missing_cancelled_or_diverted_flag")
            .when(F.col("Cancelled") == 1, "cancelled")
            .when(F.col("Diverted") == 1, "diverted")
            .when(F.col("ArrDel15").isNull(), "missing_target")
            .when(~F.col("ArrDel15").isin(0, 1), "invalid_target")
            .when((F.col("Cancelled") == 0) & (F.col("Diverted") == 0), "eligible")
            .otherwise("other_flag_value")
        )
        category_rows = flights.groupBy(category.alias("Category")).count().collect()
        category_counts = {r["Category"]: int(r["count"]) for r in category_rows}
        total_rows = sum(category_counts.values())
        log(f"Cleaned rows read: {total_rows:,}")
        order = ["cancelled", "diverted", "missing_cancelled_or_diverted_flag", "missing_target",
                 "invalid_target", "other_flag_value", "eligible"]
        eligibility_rows = [[c, category_counts.get(c, 0)] for c in order]
        for name, count in eligibility_rows:
            log(f"  {name}: {count:,}")

        eligible = keep(
            flights
            .filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) & F.col("ArrDel15").isin(0, 1))
            .join(F.broadcast(holidays), on="FlightDate", how="left")
            .withColumn("HolidayPeriod", F.coalesce(F.col("HolidayPeriod"), F.lit("Regular day")))
            .withColumn("label", F.col("ArrDel15").cast("double"))
            .withColumn("Reporting_Airline", F.coalesce(F.col("Reporting_Airline"), F.lit("UNKNOWN")))
            .withColumn("DepTimeBlk", F.coalesce(F.col("DepTimeBlk"), F.lit("UNKNOWN")))
            .withColumn("DayOfWeekCat",
                        F.coalesce(F.concat(F.lit("DOW_"), F.col("DayOfWeek").cast("string")), F.lit("DOW_UNKNOWN")))
            .withColumn("_dep_min", minute_of_day(F.col("CRSDepTime")))
            .withColumn("_arr_min", minute_of_day(F.col("CRSArrTime")))
            .withColumn("TimeParseInvalid",
                        (F.col("_dep_min").isNull() | F.col("_arr_min").isNull()).cast("int"))
            .withColumn("DepMinuteSin", F.coalesce(F.sin(F.lit(2.0 * math.pi / 1440.0) * F.col("_dep_min")), F.lit(0.0)))
            .withColumn("DepMinuteCos", F.coalesce(F.cos(F.lit(2.0 * math.pi / 1440.0) * F.col("_dep_min")), F.lit(0.0)))
            .withColumn("ArrMinuteSin", F.coalesce(F.sin(F.lit(2.0 * math.pi / 1440.0) * F.col("_arr_min")), F.lit(0.0)))
            .withColumn("ArrMinuteCos", F.coalesce(F.cos(F.lit(2.0 * math.pi / 1440.0) * F.col("_arr_min")), F.lit(0.0)))
            .withColumn("ScheduledDepHour", F.coalesce(F.floor(F.col("_dep_min") / F.lit(60.0)).cast("double"), F.lit(0.0)))
            .withColumn("ScheduledArrHour", F.coalesce(F.floor(F.col("_arr_min") / F.lit(60.0)).cast("double"), F.lit(0.0)))
            .withColumn("IsHolidayWindow", (F.col("HolidayPeriod") != F.lit("Regular day")).cast("double"))
            .select(
                "FlightDate", "label", "HolidayPeriod", "Reporting_Airline", "Origin", "Dest",
                "DepTimeBlk", "DayOfWeekCat", "CRSElapsedTime", "Distance",
                *PLAIN_NUMERIC_FEATURES, "TimeParseInvalid",
            )
        )

        # ---------------- chronological split by whole dates ----------------
        dates = [r[0] for r in eligible.select("FlightDate").distinct().orderBy("FlightDate").collect()]
        if len(dates) < 3:
            raise PipelineError("Fewer than three distinct flight dates; cannot make train/validation/test splits.")
        val_index = max(1, min(int(len(dates) * VALIDATION_START_DATE_FRACTION), len(dates) - 2))
        test_index = max(val_index + 1, min(int(len(dates) * TEST_START_DATE_FRACTION), len(dates) - 1))
        validation_start, test_start = dates[val_index], dates[test_index]
        windows = {
            "train": (dates[0], dates[val_index - 1]),
            "validation": (validation_start, dates[test_index - 1]),
            "test": (test_start, dates[-1]),
        }
        log(f"Distinct flight dates: {len(dates)}")
        for split, (start, end) in windows.items():
            log(f"{split} dates: {start} through {end}")

        split_expr = (
            F.when(F.col("FlightDate") < F.lit(validation_start), "train")
            .when(F.col("FlightDate") < F.lit(test_start), "validation")
            .otherwise("test")
        )
        accounting = (
            eligible.withColumn("Split", split_expr).groupBy("Split", "HolidayPeriod")
            .agg(F.count(F.lit(1)).alias("n"), F.sum("label").alias("pos")).collect()
        )
        split_n = {"train": 0, "validation": 0, "test": 0}
        split_pos = {"train": 0, "validation": 0, "test": 0}
        holiday_rows = []
        seen_in_fit = set()
        for r in sorted(accounting, key=lambda r: (["train", "validation", "test"].index(r["Split"]), r["HolidayPeriod"])):
            n, pos = int(r["n"]), int(r["pos"] or 0)
            split_n[r["Split"]] += n
            split_pos[r["Split"]] += pos
            holiday_rows.append([r["Split"], r["HolidayPeriod"], n, pos])
            if r["Split"] in ("train", "validation"):
                seen_in_fit.add(r["HolidayPeriod"])
        eligible_count = sum(split_n.values())
        invalid_times = int(eligible.agg(F.sum("TimeParseInvalid")).first()[0] or 0)
        for split in ("train", "validation", "test"):
            n, pos = split_n[split], split_pos[split]
            if n == 0 or pos == 0 or pos == n:
                raise PipelineError(f"Split '{split}' is empty or has only one class (n={n}, positives={pos}).")
            log(f"{split}: {n:,} flights; delayed 15+ min: {pos:,} ({100.0 * pos / n:.2f}%)")
        log(f"Eligible flights for ML: {eligible_count:,}; unparsable scheduled times (set to neutral values): {invalid_times:,}")
        train_only_cats = {r[1] for r in holiday_rows if r[0] == "train"}
        for category_name in sorted({r[1] for r in holiday_rows if r[1] != "Regular day"}):
            if category_name not in train_only_cats:
                log(f"NOTE: holiday category '{category_name}' has no flights in the TRAINING window; "
                    f"its effect cannot be learned there.")
        for category_name in sorted({r[1] for r in holiday_rows if r[0] == "test" and r[1] != "Regular day"}):
            if category_name not in seen_in_fit:
                log(f"NOTE: test holiday category '{category_name}' is never seen before the test window "
                    f"(train + validation); the model can only treat it as an unseen category.")

        train_base = eligible.filter(F.col("FlightDate") < F.lit(validation_start))
        validation_base = eligible.filter(
            (F.col("FlightDate") >= F.lit(validation_start)) & (F.col("FlightDate") < F.lit(test_start)))
        test_base = eligible.filter(F.col("FlightDate") >= F.lit(test_start))
        trainval_base = eligible.filter(F.col("FlightDate") < F.lit(test_start))

        # ---------------- stage A: fit on train, tune on validation ----------------
        stage_a = prepare_stage("STAGE A", train_base, {"validation": validation_base}, True, keep, release)
        train_vec = stage_a["fit_vec"]
        val_vec = stage_a["applied"]["validation"]
        val_prevalence = split_pos["validation"] / split_n["validation"]
        trivial_val_f1 = 2.0 * val_prevalence / (1.0 + val_prevalence)

        search_rows = []     # Model, Config, ValidationAUC_ROC, ValidationAUC_PR
        best_lr = None
        for reg in LR_REG_GRID:
            stamp(f"Logistic Regression candidate regParam={reg}")
            model = fit_logistic(train_vec, reg)
            scored = score_dataframe(model, val_vec, keep)
            roc, pr = auc_pair(scored)
            search_rows.append(["LogisticRegression", f"regParam={reg}", roc, pr])
            log(f"  validation AUC-ROC={roc:.5f} AUC-PR={pr:.5f}")
            if best_lr is None or pr > best_lr["pr"] + 1e-12 or (abs(pr - best_lr["pr"]) <= 1e-12 and reg > best_lr["reg"]):
                if best_lr is not None:
                    release(best_lr["scored"])
                best_lr = {"reg": reg, "pr": pr, "roc": roc, "scored": scored, "label": f"regParam={reg}"}
            else:
                release(scored)
        log(f"Selected Logistic Regression: {best_lr['label']} (validation AUC-PR={best_lr['pr']:.5f})")

        best_rf = None
        for config in RF_CONFIGS:
            label = f"maxDepth={config['maxDepth']},minInstancesPerNode={config['minInstancesPerNode']},numTrees={RF_NUM_TREES}"
            stamp(f"Random Forest candidate {label}")
            model = fit_forest(train_vec, config)
            scored = score_dataframe(model, val_vec, keep)
            roc, pr = auc_pair(scored)
            search_rows.append(["RandomForest", label, roc, pr])
            log(f"  validation AUC-ROC={roc:.5f} AUC-PR={pr:.5f}")
            if best_rf is None or pr > best_rf["pr"] + 1e-12:
                if best_rf is not None:
                    release(best_rf["scored"])
                best_rf = {"config": config, "pr": pr, "roc": roc, "scored": scored, "label": label}
            else:
                release(scored)
        log(f"Selected Random Forest: {best_rf['label']} (validation AUC-PR={best_rf['pr']:.5f})")

        stamp("Ablation (validation only): Logistic Regression without historical-rate features")
        ablation_model = fit_logistic(train_vec, best_lr["reg"], "features_nohist")
        ablation_scored = score_dataframe(ablation_model, val_vec, keep)
        ablation_roc, ablation_pr = auc_pair(ablation_scored)
        release(ablation_scored)
        ablation_rows = [
            ["LogisticRegression with historical-rate features", best_lr["roc"], best_lr["pr"]],
            ["LogisticRegression without historical-rate features", ablation_roc, ablation_pr],
        ]
        log(f"  validation AUC-ROC={ablation_roc:.5f} AUC-PR={ablation_pr:.5f} (without history)")

        stamp("Selecting classification thresholds on VALIDATION data only")
        lr_threshold, lr_threshold_rows = threshold_search(best_lr["scored"], THRESHOLD_CANDIDATES)
        rf_threshold, rf_threshold_rows = threshold_search(best_rf["scored"], THRESHOLD_CANDIDATES)
        lr_val_f1 = max(r[6] for r in lr_threshold_rows)
        rf_val_f1 = max(r[6] for r in rf_threshold_rows)
        log(f"Logistic Regression threshold={lr_threshold:.2f} (validation F1={lr_val_f1:.5f}); "
            f"Random Forest threshold={rf_threshold:.2f} (validation F1={rf_val_f1:.5f}); "
            f"trivial 'flag every flight' validation F1={trivial_val_f1:.5f}")
        for model_name, thr, rows_ in (("LogisticRegression", lr_threshold, lr_threshold_rows),
                                       ("RandomForest", rf_threshold, rf_threshold_rows)):
            chosen = next(r for r in rows_ if r[0] == thr)
            if chosen[5] >= 0.999 or thr <= THRESHOLD_CANDIDATES[0] or thr >= THRESHOLD_CANDIDATES[-1]:
                log(f"WARNING: {model_name} threshold {thr:.2f} is degenerate or at the grid edge "
                    f"(validation recall={chosen[5]:.3f}); interpret its F1 with care.")

        stage_a_info = {"prior": stage_a["prior"], "feature_names": stage_a["feature_names"]}
        release(best_lr["scored"])
        release(best_rf["scored"])
        release(train_vec)
        release(val_vec)
        for table in stage_a["tables"].values():
            release(table)
        del stage_a

        # ---------------- stage B: refit on train + validation, score the held-out test once ----------------
        stage_b = prepare_stage("STAGE B", trainval_base, {"test": test_base}, False, keep, release)
        final_vec = stage_b["fit_vec"]
        test_vec = stage_b["applied"]["test"]

        stamp(f"Refitting Logistic Regression ({best_lr['label']}) on train + validation")
        lr_final = fit_logistic(final_vec, best_lr["reg"])
        stamp(f"Refitting Random Forest ({best_rf['label']}) on train + validation")
        rf_final = fit_forest(final_vec, best_rf["config"])

        test_n, test_pos = split_n["test"], split_pos["test"]
        test_prevalence = test_pos / test_n
        metrics_rows, confusion_rows, ranking_rows, slice_rows = [], [], [], []
        final_models = (
            ("LogisticRegression", lr_final, lr_threshold),
            ("RandomForest", rf_final, rf_threshold),
        )
        for model_name, model, threshold in final_models:
            stamp(f"Scoring held-out test window: {model_name} (threshold {threshold:.2f} from validation)")
            scored = score_dataframe(model, test_vec, keep)
            roc, pr = auc_pair(scored)
            tp, fp, fn, tn = confusion(scored, threshold)
            precision, recall, f1, accuracy = prf(tp, fp, fn, tn)
            metrics_rows.append([model_name, float(threshold), roc, pr, f1, precision, recall, accuracy])
            for actual, predicted, count in ((0, 0, tn), (0, 1, fp), (1, 0, fn), (1, 1, tp)):
                confusion_rows.append([model_name, actual, predicted, count])
            log(f"{model_name} TEST: threshold={threshold:.2f} AUC_ROC={roc:.5f} AUC_PR={pr:.5f} "
                f"F1_class1={f1:.5f} precision_class1={precision:.5f} recall_class1={recall:.5f} accuracy={accuracy:.5f}")
            for fraction, flagged_fraction, flagged_n, p_, r_, lift in ranking_metrics(scored, TOP_FRACTIONS, test_n, test_pos):
                ranking_rows.append([model_name, fraction, flagged_fraction, flagged_n, p_, r_, lift])
            slice_rows.append([model_name, "ALL TEST", test_n, test_pos, test_prevalence, roc, pr,
                               float(threshold), precision, recall, f1, accuracy])
            for period in sorted(r[0] for r in scored.select("HolidayPeriod").distinct().collect()):
                part = scored.filter(F.col("HolidayPeriod") == period)
                n_part = part.count()
                pos_part = int(part.agg(F.sum("label")).first()[0] or 0)
                if 0 < pos_part < n_part:
                    roc_part, pr_part = auc_pair(part)
                else:
                    roc_part, pr_part = None, None
                tp_s, fp_s, fn_s, tn_s = confusion(part, threshold)
                p_s, r_s, f_s, a_s = prf(tp_s, fp_s, fn_s, tn_s)
                slice_rows.append([model_name, period, n_part, pos_part,
                                   pos_part / n_part if n_part else None, roc_part, pr_part,
                                   float(threshold), p_s, r_s, f_s, a_s])
            release(scored)

        baseline_rows = [
            ["MajorityClass_AlwaysNotDelayed", None, 0.5, test_prevalence, 0.0, 0.0, 0.0, 1.0 - test_prevalence],
            ["AlwaysDelayed", None, 0.5, test_prevalence,
             2.0 * test_prevalence / (1.0 + test_prevalence), test_prevalence, 1.0, test_prevalence],
        ]

        names = stage_b["feature_names"]
        importances = rf_final.featureImportances.toArray().tolist()
        importance_rows = sorted(
            [[names[i] if i < len(names) else f"feature_{i}", float(v)] for i, v in enumerate(importances)],
            key=lambda r: r[1], reverse=True,
        )[:30]

        # ---------------- save result tables ----------------
        stamp("Saving result tables")
        save_table(spark, "metrics.csv",
                   ["Model", "DecisionThreshold", "AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1", "Recall_class1", "Accuracy"],
                   metrics_rows)
        threshold_rows = [["LogisticRegression", *r] for r in lr_threshold_rows] + \
                         [["RandomForest", *r] for r in rf_threshold_rows]
        save_table(spark, "validation_threshold_search.csv",
                   ["Model", "Threshold", "TP", "FP", "FN", "Precision_class1", "Recall_class1", "F1_class1"],
                   threshold_rows)
        save_table(spark, "confusion_counts.csv", ["Model", "ActualLabel", "PredictedLabel", "Count"], confusion_rows)
        save_table(spark, "random_forest_feature_importance_top30.csv", ["Feature", "Importance"], importance_rows)
        class_rows = [
            [split, split_n[split], split_pos[split], split_n[split] - split_pos[split],
             windows[split][0].isoformat(), windows[split][1].isoformat()]
            for split in ("train", "validation", "test")
        ]
        save_table(spark, "class_counts.csv",
                   ["Split", "Flights", "Delayed15Plus", "NotDelayed15Plus", "StartDate", "EndDate"], class_rows)
        save_table(spark, "eligibility_counts.csv", ["Category", "Flights"], eligibility_rows)
        save_table(spark, "holiday_category_counts.csv", ["Split", "HolidayPeriod", "Flights", "Delayed15Plus"], holiday_rows)
        selected_label = {"LogisticRegression": best_lr["label"], "RandomForest": best_rf["label"]}
        save_table(spark, "hyperparameter_search.csv",
                   ["Model", "Config", "ValidationAUC_ROC", "ValidationAUC_PR", "Selected"],
                   [[r[0], r[1], r[2], r[3], "yes" if r[1] == selected_label[r[0]] else "no"] for r in search_rows])
        save_table(spark, "ablation_validation_metrics.csv",
                   ["Variant", "ValidationAUC_ROC", "ValidationAUC_PR"], ablation_rows)
        save_table(spark, "baseline_metrics.csv",
                   ["Model", "DecisionThreshold", "AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1", "Recall_class1", "Accuracy"],
                   baseline_rows)
        save_table(spark, "ranking_metrics.csv",
                   ["Model", "TargetFlaggedFraction", "ActualFlaggedFraction", "FlaggedFlights", "Precision_class1", "Recall_class1", "Lift"],
                   ranking_rows)
        save_table(spark, "metrics_by_holiday_slice.csv",
                   ["Model", "Slice", "Flights", "Delayed15Plus", "Prevalence", "AUC_ROC", "AUC_PR",
                    "DecisionThreshold", "Precision_class1", "Recall_class1", "F1_class1", "Accuracy"],
                   slice_rows)

        # ---------------- save models, preprocessing and metadata ----------------
        stamp("Saving fitted preprocessing pipeline, classifiers and historical-rate tables")
        stage_b["pipeline_model"].write().overwrite().save(f"{HDFS_MODELS}/feature_pipeline")
        lr_final.write().overwrite().save(f"{HDFS_MODELS}/logistic_regression")
        rf_final.write().overwrite().save(f"{HDFS_MODELS}/random_forest")
        for table_name, table in stage_b["tables"].items():
            table.write.mode("overwrite").parquet(f"{HDFS_MODELS}/historical_rate_tables/{table_name}")
        metadata = {
            "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "spark_version": spark.version,
            "target": "ArrDel15 (1 = arrival delay >= 15 minutes); cancelled/diverted excluded",
            "windows": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in windows.items()},
            "final_fit_window": [windows["train"][0].isoformat(), windows["validation"][1].isoformat()],
            "feature_source_columns": FEATURE_SOURCE_COLUMNS,
            "features": names,
            "history_keys": HISTORY_KEYS,
            "history_smoothing": HISTORY_SMOOTHING,
            "history_prior_stage_a_train": stage_a_info["prior"],
            "history_prior_final_fit": stage_b["prior"],
            "selected_logistic_regression": best_lr["label"],
            "selected_random_forest": best_rf["label"],
            "thresholds_from_validation": {"LogisticRegression": lr_threshold, "RandomForest": rf_threshold},
            "random_forest": {"numTrees": RF_NUM_TREES, "maxBins": RF_MAX_BINS, "subsamplingRate": RF_SUBSAMPLING_RATE,
                              "featureSubsetStrategy": RF_FEATURE_SUBSET},
            "logistic_regression": {"maxIter": LR_MAX_ITER, "elasticNetParam": 0.0},
            "seed": SEED,
        }
        metadata_text = json.dumps(metadata, indent=2)
        with open(os.path.join(LOCAL_RESULTS, "preprocessing_metadata.json"), "w", encoding="utf-8") as handle:
            handle.write(metadata_text + "\n")
        save_text_hdfs(spark, "preprocessing_metadata", metadata_text)
        log(f"Saved models under {HDFS_MODELS}")

        # ---------------- run summary ----------------
        summary_path = os.path.join(LOCAL_RESULTS, "run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as s:
            s.write("BDA MLlib flight-arrival-delay classification (improved script)\n")
            s.write("Target: ArrDel15; positive class: arrival delay >= 15 minutes\n")
            s.write("Cancelled and diverted flights excluded; valid 0/1 target required.\n")
            s.write("Predictors: pre-departure schedule, carrier, cyclic time features, holiday calendar and\n")
            s.write("  smoothed historical delay rates (strictly earlier dates for fitting rows).\n")
            s.write("No post-departure or delay-cause columns, no Tail_Number, no raw Year/Month/DayOfYear.\n")
            s.write(f"Eligible records: {eligible_count}\n")
            s.write(f"Distinct dates: {len(dates)}\n")
            s.write(f"Training date range: {windows['train'][0]} through {windows['train'][1]}\n")
            s.write(f"Validation date range (tuning and threshold selection): {windows['validation'][0]} through {windows['validation'][1]}\n")
            s.write(f"Final test date range (held out): {windows['test'][0]} through {windows['test'][1]}\n")
            for split in ("train", "validation", "test"):
                s.write(f"{split.capitalize()} records: {split_n[split]}; positive: {split_pos[split]} "
                        f"({100.0 * split_pos[split] / split_n[split]:.2f}%)\n")
            s.write(f"Final models refitted on train + validation: {split_n['train'] + split_n['validation']} records\n")
            s.write(f"Selected Logistic Regression: {best_lr['label']} (maxIter={LR_MAX_ITER})\n")
            s.write(f"Selected Random Forest: {best_rf['label']} (maxBins={RF_MAX_BINS}, "
                    f"subsamplingRate={RF_SUBSAMPLING_RATE}, featureSubsetStrategy={RF_FEATURE_SUBSET})\n")
            s.write("Hyperparameters chosen by validation AUC-PR; thresholds chosen by validation F1 only.\n")
            s.write(f"Trivial references on test: majority class accuracy={1.0 - test_prevalence:.4f}; "
                    f"'flag every flight' F1={2.0 * test_prevalence / (1.0 + test_prevalence):.4f}; "
                    f"random-ranking AUC-PR equals prevalence={test_prevalence:.4f}.\n")
            s.write(f"Validation-only ablation (LR) AUC-PR: with history={best_lr['pr']:.5f}, without history={ablation_pr:.5f}\n")
            s.write("AUC-ROC and AUC-PR use the class-1 probability; F1, precision and recall refer to class 1\n")
            s.write("at the validation-selected threshold. Holiday categories absent from the fitting windows\n")
            s.write("are treated as unseen categories. Results are observational, not causal.\n\n")
            s.write("Model, DecisionThreshold, AUC_ROC, AUC_PR, F1_class1, Precision_class1, Recall_class1, Accuracy\n")
            for row in metrics_rows:
                s.write(", ".join([row[0]] + [f"{value:.6f}" for value in row[1:]]) + "\n")
        log(f"Local run summary: {summary_path}")

        log(f"ML_ELIGIBLE_ROWS={eligible_count}")
        log(f"ML_TRAIN_ROWS={split_n['train']}")
        log(f"ML_VALIDATION_ROWS={split_n['validation']}")
        log(f"ML_TEST_ROWS={split_n['test']}")
        log(f"ML_FINAL_FIT_ROWS={split_n['train'] + split_n['validation']}")
        log(f"ML_TEST_START_DATE={test_start.isoformat()}")
        log("MLLIB_JOB_STATUS=SUCCEEDED")
        log(f"MLLIB_FINISHED_UTC={datetime.utcnow().isoformat(timespec='seconds')}Z")
        return 0
    except PipelineError as error:
        log(f"ERROR: {error}")
        log("MLLIB_JOB_STATUS=FAILED")
        return 2
    except Exception as error:
        log(f"ERROR: MLlib pipeline failed: {type(error).__name__}: {error}")
        log(traceback.format_exc())
        log("MLLIB_JOB_STATUS=FAILED")
        return 1
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist()
            except Exception:
                pass
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())