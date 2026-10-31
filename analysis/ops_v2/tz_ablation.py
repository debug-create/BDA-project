#!/usr/bin/env python3
"""Prepare and evaluate the four validation-only UTC-correction ablations."""

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

SRC = "/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import asof_ops
import asof_ops_utc

spec = importlib.util.spec_from_file_location("ml_stage", "/src/03_ml.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


CLEAN = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
ROOT_HDFS = "hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/tz_ablation"
ROOT_LOCAL = "/results/experiments/ml_redesign/ops_v2/tz_ablation"
PREPARED = f"{ROOT_HDFS}/prepared"
OLD_FEATURES = f"{ROOT_HDFS}/old_features"
LEAD = 120


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "model"], required=True)
    parser.add_argument("--reuse-old", action="store_true",
                        help="reuse the already materialized old-clock feature checkpoint")
    return parser.parse_args()


def local_write(name, header, rows):
    os.makedirs(ROOT_LOCAL, exist_ok=True)
    with open(os.path.join(ROOT_LOCAL, name), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header); writer.writerows(rows)


def main():
    args = parse_args()
    spark = (SparkSession.builder.appName(f"BDA-OpsV2-TZ-{args.phase}")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.ui.retainedJobs", "40")
             .config("spark.ui.retainedStages", "80")
             .config("spark.ui.retainedTasks", "1000")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    cached = []

    storage_level = StorageLevel.DISK_ONLY if args.phase == "model" else StorageLevel.MEMORY_AND_DISK

    def keep(frame):
        # The model input is deliberately wide (old and UTC variants side by
        # side).  With 512 MiB executors, caching it in JVM memory can push the
        # YARN container over its physical limit before fitting even starts.
        frame.persist(storage_level); cached.append(frame); return frame

    def release(frame):
        try:
            frame.unpersist()
        except Exception:
            pass
        cached[:] = [item for item in cached if item is not frame]

    try:
        if args.phase == "prepare":
            prepare(spark, keep, release, args.reuse_old)
        else:
            model(spark, keep, release)
        print(f"TZ_ABLATION_{args.phase.upper()}_STATUS=SUCCEEDED", flush=True)
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"TZ_ABLATION_{args.phase.upper()}_STATUS=FAILED error={type(exc).__name__}: {exc}", flush=True)
        return 2
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist()
            except Exception:
                pass
        spark.stop()


def inputs(spark):
    flights = spark.read.parquet(CLEAN)
    holidays = (spark.read.option("header", True).csv(HOLIDAYS)
                .select("FlightDate", "HolidayPeriod")
                .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
                .filter(F.col("FlightDate").isNotNull()))
    dates = [row[0] for row in flights.select("FlightDate").distinct().orderBy("FlightDate").collect()]
    windows = ml.compute_windows(dates)
    return flights, holidays, windows


def build_eligible(selection_source, holidays, keep):
    demand = keep(ml.compute_demand_table(selection_source))
    eligible = keep(ml.build_eligible_frame(selection_source, holidays, demand))
    eligible.count()
    return demand, eligible


def prepare(spark, keep, release, reuse_old=False):
    started = time.time()
    flights, holidays, windows = inputs(spark)
    validation_start, test_start = windows["validation"][0], windows["test"][0]
    selection_source = flights.filter(F.col("FlightDate") < F.lit(test_start))
    expected = selection_source.filter(ml.eligible_condition()).count()
    timezone_table = keep(asof_ops_utc.load_timezone_offsets(spark)); timezone_table.count()

    old_priors = None
    if reuse_old:
        old_group_count = spark.read.parquet(OLD_FEATURES).count()
        print(f"REUSING_OLD_FEATURE_CHECKPOINT rows={old_group_count}", flush=True)
    else:
        demand, eligible = build_eligible(selection_source, holidays, keep)
        old_full, old_priors = asof_ops.attach_asof_features(
            selection_source, eligible, LEAD, validation_start, keep, release)
        old_groups = (old_full.select(
            *asof_ops.JOIN_KEYS,
            *[F.col(name).alias(f"Old_{name}") for name in asof_ops.OPS_FEATURES])
            .dropDuplicates(asof_ops.JOIN_KEYS))
        old_groups.write.mode("overwrite").parquet(OLD_FEATURES)
        old_group_count = old_groups.count()
        release(old_full); release(eligible); release(demand)

    demand, eligible = build_eligible(selection_source, holidays, keep)
    spark.sparkContext.setCheckpointDir(f"{ROOT_HDFS}/checkpoints")
    utc_full, utc_priors, coverage = asof_ops_utc.attach_asof_features_utc(
        selection_source, eligible, LEAD, validation_start, timezone_table, keep, release)
    base_columns = [column for column in utc_full.columns if column not in asof_ops.OPS_FEATURES]
    utc_named = utc_full.select(
        *base_columns, *[F.col(name).alias(f"Utc_{name}") for name in asof_ops.OPS_FEATURES])
    old_read = spark.read.parquet(OLD_FEATURES)
    prepared = utc_named.join(old_read, asof_ops.JOIN_KEYS, "left")
    actual = prepared.count()
    if actual != expected:
        raise ValueError(f"prepared row count changed {expected:,} -> {actual:,}")
    null_old = prepared.agg(*[F.sum(F.col(f"Old_{name}").isNull().cast("long")).alias(name)
                              for name in asof_ops.OPS_FEATURES]).first().asDict()
    if any(int(value or 0) for value in null_old.values()):
        raise ValueError(f"old feature join contains nulls: {null_old}")
    prepared.write.mode("overwrite").parquet(PREPARED)
    metadata = {
        "status": "validation-only; test rows excluded before eligibility/outcome processing",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lead_minutes": LEAD,
        "windows": {name: [str(bounds[0]), str(bounds[1])] for name, bounds in windows.items()},
        "eligible_pretest_rows": expected,
        "prepared_rows": actual,
        "old_feature_groups": old_group_count,
        "timezone_coverage": coverage,
        "old_priors": old_priors,
        "utc_priors": utc_priors,
        "runtime_seconds": time.time() - started,
    }
    os.makedirs(ROOT_LOCAL, exist_ok=True)
    with open(os.path.join(ROOT_LOCAL, "prepare_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2); handle.write("\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)


def e13_spec():
    return {"name": "E13_e6_plus_ops", "factor": "UTC ablation", "use_history": True,
            "window_days": None, "warmup_days": 7, "use_holiday": True, "use_demand": False,
            "history_set": "extended", "recent_days": 7, "extra_demand": True,
            "sched_extras": False, "asof_ops": False}


def make_pipeline(specification, ops_columns):
    impute, plain, history, categorical = ml.spec_columns(specification)
    plain = list(plain) + list(ops_columns)
    imputed = [f"{column}__imputed" for column in impute]
    indices = [f"{column}__idx" for column in categorical]
    encoded = [f"{column}__ohe" for column in categorical]
    stages = [Imputer(strategy="median", inputCols=impute, outputCols=imputed)]
    stages.extend(StringIndexer(inputCol=column, outputCol=index, handleInvalid="keep")
                  for column, index in zip(categorical, indices))
    stages.append(OneHotEncoder(inputCols=indices, outputCols=encoded, handleInvalid="keep", dropLast=False))
    stages.append(VectorAssembler(inputCols=imputed + plain + history + encoded,
                                  outputCol="features", handleInvalid="keep"))
    return Pipeline(stages=stages)


def model(spark, keep, release):
    started = time.time()
    # Stream the materialized source instead of caching all old+new columns.
    # Reusable, narrower derivatives below are persisted DISK_ONLY.
    prepared = spark.read.parquet(PREPARED)
    dates = [row[0] for row in prepared.select("FlightDate").distinct().orderBy("FlightDate").collect()]
    # Prepared contains train+validation only, so use the known split boundary.
    validation_start = datetime.strptime("2024-11-25", "%Y-%m-%d").date()
    train_base = prepared.filter(F.col("FlightDate") < F.lit(validation_start))
    validation_base = prepared.filter(F.col("FlightDate") >= F.lit(validation_start))
    train_n, validation_n = train_base.count(), validation_base.count()
    specification = e13_spec()

    groups = ml.history_groups(specification)
    train_all = train_base
    val_all = validation_base
    history_tables = []
    for group in groups:
        prior, tables = ml.build_history_tables(train_base, keep, group["window_days"], group["keys_map"])
        history_tables.extend(tables.values())
        train_all = ml.apply_history_training_past_only(train_all, group["window_days"], group["keys_map"])
        val_all = ml.apply_history(val_all, prior, tables, group["keys_map"])
    train_all = keep(train_all); val_all = keep(val_all)
    ml.check_rows("UTC ablation history train", train_n, train_all)
    ml.check_rows("UTC ablation history validation", validation_n, val_all)
    first_date = train_base.agg(F.min("FlightDate")).first()[0]
    train_fit = keep(train_all.filter(F.col("FlightDate") >= F.date_add(F.lit(first_date), 7)))
    train_fit.count()

    variants = [
        ("a_origin_only_utc", [f"Utc_{name}" for name in asof_ops_utc.ORIGIN_ONLY_FEATURES]),
        ("b_origin_plus_inbound_utc", [f"Utc_{name}" for name in asof_ops_utc.ORIGIN_INBOUND_FEATURES]),
        ("c_all_old_local_clock", [f"Old_{name}" for name in asof_ops.OPS_FEATURES]),
        ("d_all_new_utc", [f"Utc_{name}" for name in asof_ops.OPS_FEATURES]),
    ]
    metrics = []
    histograms = []
    for name, ops_columns in variants:
        variant_started = time.time()
        pipeline_model = make_pipeline(specification, ops_columns).fit(train_fit)
        train_vec = keep(pipeline_model.transform(train_fit).select("FlightDate", "label", "HolidayPeriod", "features"))
        val_vec = keep(pipeline_model.transform(val_all).select("FlightDate", "label", "HolidayPeriod", "features"))
        train_vec.count(); val_vec.count()
        lr = ml.fit_logistic(train_vec, 0.001)
        scored = keep(lr.transform(val_vec).select(
            "FlightDate", "label", "probability",
            vector_to_array("probability")[1].alias("score")))
        scored.count()
        roc, pr = ml.auc_pair(scored)
        ranking = ml.ranking_metrics(scored, [0.10], validation_n,
                                     int(validation_base.agg(F.sum("label")).first()[0]))[0]
        _, actual_fraction, flagged, precision, recall, lift = ranking
        metrics.append([name, len(ops_columns), roc, pr, actual_fraction, flagged,
                        precision, recall, lift, time.time() - variant_started])
        rows = (scored.select("FlightDate", "label",
                              F.floor(F.col("score") * 1000).cast("int").alias("ScoreBin"))
                .groupBy("FlightDate", "ScoreBin")
                .agg(F.count("*").alias("Flights"), F.sum("label").cast("long").alias("Delayed15Plus"))
                .collect())
        histograms.extend([name, str(row["FlightDate"]), int(row["ScoreBin"]),
                           int(row["Flights"]), int(row["Delayed15Plus"] or 0)] for row in rows)
        print(f"TZ_ABLATION_METRIC variant={name} roc={roc:.8f} pr={pr:.8f} "
              f"top10_precision={precision:.8f} lift={lift:.8f}", flush=True)
        release(scored); release(train_vec); release(val_vec)

    local_write("validation_metrics.csv",
                ["Variant", "OpsFeatureCount", "ValidationAUC_ROC", "ValidationAUC_PR",
                 "ActualTop10Fraction", "FlaggedFlights", "Top10Precision", "Top10Recall", "Top10Lift",
                 "RuntimeSeconds"], metrics)
    local_write("validation_score_histograms_1000bin.csv",
                ["Variant", "FlightDate", "ScoreBin", "Flights", "Delayed15Plus"], histograms)
    metadata = {"status": "VALIDATION ONLY - test window never read", "train_rows": train_n,
                "validation_rows": validation_n, "fitting_rows_after_warmup": train_fit.count(),
                "lr_reg_param": 0.001, "lead_minutes": LEAD, "runtime_seconds": time.time() - started,
                "prepared_dates": [str(min(dates)), str(max(dates))]}
    with open(os.path.join(ROOT_LOCAL, "model_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2); handle.write("\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
