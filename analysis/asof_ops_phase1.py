#!/usr/bin/env python3
"""Phase-1 feasibility test for conservative as-of operational features.

The job reads only dates before the frozen test window.  Every operational
event is reduced to an integer local-airport hour and a target flight may use
only event hours strictly before its prediction-time hour.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from datetime import date

from pyspark import StorageLevel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F


ROOT = "/src"
CLEAN = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
FINAL_LOCAL = "/results/experiments/ml_redesign/actual_final"
OUT_LOCAL = "/results/experiments/ml_redesign/asof_ops"
OUT_HDFS = "hdfs://master:9000/user/root/flights/experiments/ml_redesign/asof_ops"
TEST_START = "2024-12-04"
VALIDATION_START = "2024-11-25"
GATE_REFERENCE_PR = 0.2422
GATE_GAIN = 0.010
SMOOTHING = 20.0

OPS_FEATURES = [
    "OpsOriginDepartedCount3h", "OpsOriginMeanDepDelay3h", "OpsOriginDepDelay15Share3h",
    "OpsOriginBacklogCount3h", "OpsOriginBacklogShare3h",
    "OpsOriginInboundLandedCount3h", "OpsOriginInboundMeanArrDelay3h", "OpsOriginInboundArrDelay15Share3h",
    "OpsCarrierDepartedCountToday", "OpsCarrierMeanDepDelayToday", "OpsCarrierDepDelay15ShareToday",
    "OpsCarrierScheduledBeforeCountToday", "OpsCarrierBacklogShareToday",
    "OpsDestLandedCount3h", "OpsDestMeanArrDelay3h", "OpsDestArrDelay15Share3h",
]
JOIN_KEYS = ["FlightDate", "Origin", "Dest", "Reporting_Airline", "ScheduledDepHour"]


def load_ml():
    spec = importlib.util.spec_from_file_location("flight_ml", os.path.join(ROOT, "03_ml.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-minutes", type=int, choices=[60, 120, 180], default=120)
    parser.add_argument("--mode", choices=["prepare", "model", "all"], default="all")
    return parser.parse_args()


def write_csv(spark, name, header, rows, lead):
    os.makedirs(OUT_LOCAL, exist_ok=True)
    path = os.path.join(OUT_LOCAL, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([["" if value is None else value for value in row] for row in rows])
    schema = ",".join(f"`{column}` string" for column in header)
    values = [["" if value is None else str(value) for value in row] for row in rows]
    (spark.createDataFrame(values, schema).coalesce(1).write.mode("overwrite")
     .option("header", True).csv(f"{OUT_HDFS}/lead_{lead}/{name.removesuffix('.csv')}"))
    print(f"WROTE {path}", flush=True)


def empty_int_array():
    return F.array().cast("array<int>")


def sequence_if(start, end, condition):
    return F.when(condition, F.sequence(start.cast("int"), end.cast("int"))).otherwise(empty_int_array())


def check_count(label, expected, frame):
    actual = frame.count()
    if actual != expected:
        raise RuntimeError(f"{label}: row count changed from {expected:,} to {actual:,}")
    print(f"ROW_COUNT_CHECK=PASSED {label} rows={actual}", flush=True)


def add_time_columns(flights, ml):
    dep_min = ml.minute_of_day(F.col("CRSDepTime"))
    arr_min = ml.minute_of_day(F.col("CRSArrTime"))
    day = F.datediff("FlightDate", F.lit("1970-01-01").cast("date"))
    sched_dep_abs_min = day * F.lit(1440) + dep_min
    sched_arr_abs_min = (day + F.when(arr_min < dep_min, 1).otherwise(0)) * F.lit(1440) + arr_min
    return (flights
            .withColumn("_dep_min", dep_min)
            .withColumn("_arr_min", arr_min)
            .withColumn("_sched_dep_abs_hour", F.floor(sched_dep_abs_min / 60).cast("long"))
            .withColumn("_actual_dep_abs_hour",
                        F.floor((sched_dep_abs_min + F.col("DepDelay")) / 60).cast("long"))
            .withColumn("_actual_arr_abs_hour",
                        F.floor((sched_arr_abs_min + F.col("ArrDelay")) / 60).cast("long")))


def aggregate_priors(events):
    train = events.filter(F.col("FlightDate") < F.lit(VALIDATION_START))
    valid_dep = train.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) & F.col("DepDelay").isNotNull())
    valid_arr = train.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) & F.col("ArrDelay").isNotNull())
    dep = valid_dep.agg(F.avg("DepDelay").alias("mean"),
                        F.avg((F.col("DepDelay") >= 15).cast("double")).alias("share")).first()
    arr = valid_arr.agg(F.avg("ArrDelay").alias("mean"),
                        F.avg((F.col("ArrDelay") >= 15).cast("double")).alias("share")).first()
    return {
        "dep_mean": float(dep["mean"]), "dep_share": float(dep["share"]),
        "arr_mean": float(arr["mean"]), "arr_share": float(arr["share"]),
    }


def departed_window(events):
    valid = events.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
        F.col("DepDelay").isNotNull() & F.col("_actual_dep_abs_hour").isNotNull())
    hourly = (valid.groupBy(F.col("Origin").alias("_airport"),
                            F.col("_actual_dep_abs_hour").alias("_event_hour"))
              .agg(F.count("*").alias("_dep_n"), F.sum("DepDelay").alias("_dep_sum"),
                   F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_dep15")))
    return (hourly.withColumn("_t_abs", F.explode(F.sequence(F.col("_event_hour") + 1,
                                                              F.col("_event_hour") + 3)))
            .groupBy("_airport", "_t_abs")
            .agg(F.sum("_dep_n").alias("_dep_n"), F.sum("_dep_sum").alias("_dep_sum"),
                 F.sum("_dep15").alias("_dep15")))


def landed_window(events):
    valid = events.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
        F.col("ArrDelay").isNotNull() & F.col("_actual_arr_abs_hour").isNotNull())
    hourly = (valid.groupBy(F.col("Dest").alias("_airport"),
                            F.col("_actual_arr_abs_hour").alias("_event_hour"))
              .agg(F.count("*").alias("_arr_n"), F.sum("ArrDelay").alias("_arr_sum"),
                   F.sum((F.col("ArrDelay") >= 15).cast("long")).alias("_arr15")))
    return (hourly.withColumn("_t_abs", F.explode(F.sequence(F.col("_event_hour") + 1,
                                                              F.col("_event_hour") + 3)))
            .groupBy("_airport", "_t_abs")
            .agg(F.sum("_arr_n").alias("_arr_n"), F.sum("_arr_sum").alias("_arr_sum"),
                 F.sum("_arr15").alias("_arr15")))


def origin_backlog_window(events):
    expanded = events.select(
        F.col("Origin").alias("_airport"), "Cancelled", "Diverted", "_actual_dep_abs_hour",
        F.explode(F.sequence(F.col("_sched_dep_abs_hour") + 1,
                             F.col("_sched_dep_abs_hour") + 3)).alias("_t_abs"),
    )
    backlog = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
               (F.col("_actual_dep_abs_hour") >= F.col("_t_abs")))
    return (expanded.groupBy("_airport", "_t_abs")
            .agg(F.count("*").alias("_sched_n"),
                 F.sum(backlog.cast("long")).alias("_backlog_n")))


def carrier_departed_today(events):
    valid = events.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
        F.col("DepDelay").isNotNull() & F.col("_actual_dep_abs_hour").isNotNull())
    event_day = F.floor(F.col("_actual_dep_abs_hour") / 24).cast("long")
    event_hour = F.pmod(F.col("_actual_dep_abs_hour"), F.lit(24)).cast("int")
    hourly = (valid.groupBy("Reporting_Airline", event_day.alias("_t_day"), event_hour.alias("_event_hod"))
              .agg(F.count("*").alias("_dep_n"), F.sum("DepDelay").alias("_dep_sum"),
                   F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_dep15")))
    expanded = hourly.withColumn(
        "_t_hod", F.explode(sequence_if(F.col("_event_hod") + 1, F.lit(23), F.col("_event_hod") < 23)))
    return (expanded.groupBy("Reporting_Airline", "_t_day", "_t_hod")
            .agg(F.sum("_dep_n").alias("_carrier_dep_n"), F.sum("_dep_sum").alias("_carrier_dep_sum"),
                 F.sum("_dep15").alias("_carrier_dep15")))


def carrier_backlog_today(events):
    sched_day = F.floor(F.col("_sched_dep_abs_hour") / 24).cast("long")
    sched_hod = F.pmod(F.col("_sched_dep_abs_hour"), F.lit(24)).cast("int")
    base = events.select("Reporting_Airline", "Cancelled", "Diverted", "_actual_dep_abs_hour",
                         sched_day.alias("_t_day"), sched_hod.alias("_sched_hod"))
    expanded = base.withColumn(
        "_t_hod", F.explode(sequence_if(F.col("_sched_hod") + 1, F.lit(23), F.col("_sched_hod") < 23)))
    snapshot_abs = F.col("_t_day") * 24 + F.col("_t_hod")
    backlog = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
               (F.col("_actual_dep_abs_hour") >= snapshot_abs))
    return (expanded.groupBy("Reporting_Airline", "_t_day", "_t_hod")
            .agg(F.count("*").alias("_carrier_sched_n"),
                 F.sum(backlog.cast("long")).alias("_carrier_backlog_n")))


def add_block(targets, block, condition, columns, label):
    expected = targets.count()
    joined = targets.join(block, condition, "left").drop(block["_t_abs"]) if "_t_abs" in block.columns else targets.join(block, condition, "left")
    check_count(label, expected, joined)
    return joined.select(*targets.columns, *[F.col(column) for column in columns])


def build_ops_features(events, eligible, lead_minutes, priors, keep, release):
    lead_hours = lead_minutes // 60
    targets = (eligible.select(*JOIN_KEYS).distinct()
               .withColumn("_flight_day", F.datediff("FlightDate", F.lit("1970-01-01").cast("date")))
               .withColumn("_t_abs", F.col("_flight_day") * 24 + F.col("ScheduledDepHour").cast("long") - lead_hours)
               .withColumn("_t_day", F.floor(F.col("_t_abs") / 24).cast("long"))
               .withColumn("_t_hod", F.pmod(F.col("_t_abs"), F.lit(24)).cast("int")))
    targets = keep(targets)
    targets.count()

    departure_events = events.filter(F.col("DepDelay").isNull() | (F.col("DepDelay") > -lead_minutes))
    dep = keep(departed_window(departure_events))
    before = targets.count()
    joined = targets.join(
        dep, (targets["Origin"] == dep["_airport"]) & (targets["_t_abs"] == dep["_t_abs"]), "left")
    check_count("origin departed 3h join", before, joined)
    targets2 = keep(joined.select(*[targets[column] for column in targets.columns],
                                  dep["_dep_n"], dep["_dep_sum"], dep["_dep15"]))
    targets2.count()
    release(targets); release(dep); targets = targets2

    backlog = keep(origin_backlog_window(events))
    before = targets.count()
    joined = targets.join(
        backlog, (targets["Origin"] == backlog["_airport"]) & (targets["_t_abs"] == backlog["_t_abs"]), "left")
    check_count("origin backlog 3h join", before, joined)
    targets2 = keep(joined.select(*[targets[column] for column in targets.columns],
                                  backlog["_sched_n"], backlog["_backlog_n"]))
    targets2.count()
    release(targets); release(backlog); targets = targets2

    arrival_events = events.filter(F.col("_actual_arr_abs_hour").isNull() |
                                   (F.col("_actual_arr_abs_hour") >= F.col("_sched_dep_abs_hour")))
    arrivals = keep(landed_window(arrival_events))
    for airport_column, prefix in (("Origin", "_origin_arr"), ("Dest", "_dest_arr")):
        renamed = arrivals.select(
            F.col("_airport").alias(f"{prefix}_airport"), F.col("_t_abs").alias(f"{prefix}_t_abs"),
            F.col("_arr_n").alias(f"{prefix}_n"), F.col("_arr_sum").alias(f"{prefix}_sum"),
            F.col("_arr15").alias(f"{prefix}_15"))
        before = targets.count()
        joined = targets.join(
            renamed, (targets[airport_column] == renamed[f"{prefix}_airport"]) &
            (targets["_t_abs"] == renamed[f"{prefix}_t_abs"]), "left")
        check_count(f"{airport_column.lower()} landed 3h join", before, joined)
        targets2 = keep(joined.select(*targets.columns, f"{prefix}_n", f"{prefix}_sum", f"{prefix}_15"))
        targets2.count()
        release(targets); targets = targets2
    release(arrivals)

    carrier_dep = keep(carrier_departed_today(departure_events))
    before = targets.count()
    joined = targets.join(carrier_dep, ["Reporting_Airline", "_t_day", "_t_hod"], "left")
    check_count("carrier departed today join", before, joined)
    targets2 = keep(joined.select(*targets.columns, "_carrier_dep_n", "_carrier_dep_sum", "_carrier_dep15"))
    targets2.count()
    release(targets); release(carrier_dep); targets = targets2

    carrier_backlog = keep(carrier_backlog_today(events))
    before = targets.count()
    joined = targets.join(carrier_backlog, ["Reporting_Airline", "_t_day", "_t_hod"], "left")
    check_count("carrier backlog today join", before, joined)
    targets2 = keep(joined.select(*targets.columns, "_carrier_sched_n", "_carrier_backlog_n"))
    targets2.count()
    release(targets); release(carrier_backlog); targets = targets2

    def smooth_sum(sum_col, n_col, prior):
        return ((F.coalesce(F.col(sum_col), F.lit(0.0)) + F.lit(SMOOTHING * prior)) /
                (F.coalesce(F.col(n_col), F.lit(0.0)) + F.lit(SMOOTHING)))

    def smooth_share(pos_col, n_col, prior):
        return ((F.coalesce(F.col(pos_col), F.lit(0.0)) + F.lit(SMOOTHING * prior)) /
                (F.coalesce(F.col(n_col), F.lit(0.0)) + F.lit(SMOOTHING)))

    # Backlog prior is measured from the training-date snapshot aggregates only.
    train_targets = targets.filter(F.col("FlightDate") < F.lit(VALIDATION_START))
    backlog_totals = train_targets.agg(
        F.sum(F.coalesce("_backlog_n", F.lit(0))).alias("b"),
        F.sum(F.coalesce("_sched_n", F.lit(0))).alias("n"),
        F.sum(F.coalesce("_carrier_backlog_n", F.lit(0))).alias("cb"),
        F.sum(F.coalesce("_carrier_sched_n", F.lit(0))).alias("cn")).first()
    origin_backlog_prior = float(backlog_totals["b"] or 0) / max(float(backlog_totals["n"] or 0), 1.0)
    carrier_backlog_prior = float(backlog_totals["cb"] or 0) / max(float(backlog_totals["cn"] or 0), 1.0)
    priors = dict(priors, origin_backlog_share=origin_backlog_prior,
                  carrier_backlog_share=carrier_backlog_prior, smoothing=SMOOTHING)

    out = targets.select(
        *JOIN_KEYS,
        F.coalesce("_dep_n", F.lit(0)).cast("double").alias("OpsOriginDepartedCount3h"),
        smooth_sum("_dep_sum", "_dep_n", priors["dep_mean"]).alias("OpsOriginMeanDepDelay3h"),
        smooth_share("_dep15", "_dep_n", priors["dep_share"]).alias("OpsOriginDepDelay15Share3h"),
        F.coalesce("_backlog_n", F.lit(0)).cast("double").alias("OpsOriginBacklogCount3h"),
        smooth_share("_backlog_n", "_sched_n", priors["origin_backlog_share"]).alias("OpsOriginBacklogShare3h"),
        F.coalesce("_origin_arr_n", F.lit(0)).cast("double").alias("OpsOriginInboundLandedCount3h"),
        smooth_sum("_origin_arr_sum", "_origin_arr_n", priors["arr_mean"]).alias("OpsOriginInboundMeanArrDelay3h"),
        smooth_share("_origin_arr_15", "_origin_arr_n", priors["arr_share"]).alias("OpsOriginInboundArrDelay15Share3h"),
        F.coalesce("_carrier_dep_n", F.lit(0)).cast("double").alias("OpsCarrierDepartedCountToday"),
        smooth_sum("_carrier_dep_sum", "_carrier_dep_n", priors["dep_mean"]).alias("OpsCarrierMeanDepDelayToday"),
        smooth_share("_carrier_dep15", "_carrier_dep_n", priors["dep_share"]).alias("OpsCarrierDepDelay15ShareToday"),
        F.coalesce("_carrier_sched_n", F.lit(0)).cast("double").alias("OpsCarrierScheduledBeforeCountToday"),
        smooth_share("_carrier_backlog_n", "_carrier_sched_n", priors["carrier_backlog_share"]).alias("OpsCarrierBacklogShareToday"),
        F.coalesce("_dest_arr_n", F.lit(0)).cast("double").alias("OpsDestLandedCount3h"),
        smooth_sum("_dest_arr_sum", "_dest_arr_n", priors["arr_mean"]).alias("OpsDestMeanArrDelay3h"),
        smooth_share("_dest_arr_15", "_dest_arr_n", priors["arr_share"]).alias("OpsDestArrDelay15Share3h"),
    )
    return out, priors


def auc_pair(scored):
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score")
    roc = evaluator.setMetricName("areaUnderROC").evaluate(scored)
    pr = evaluator.setMetricName("areaUnderPR").evaluate(scored)
    return float(roc), float(pr)


def score_model(ml, train, validation, spec, keep, release):
    stage = ml.prepare_stage("phase1", spec, train, {"validation": validation}, keep, release)
    model = ml.fit_logistic(stage["fit_vec"], 0.001)
    scored = keep(model.transform(stage["applied"]["validation"]).select(
        "label", vector_to_array("probability")[1].alias("score")))
    scored.count()
    roc, pr = auc_pair(scored)
    agg = scored.agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
    top = ml.ranking_metrics(scored, [0.10], int(agg["n"]), int(agg["p"]))[0]
    result = {"roc": roc, "pr": pr, "top10_precision": top[3], "top10_recall": top[4], "top10_lift": top[5],
              "rows": int(agg["n"]), "positives": int(agg["p"]), "features": len(stage["feature_names"])}
    release(scored)
    ml.release_stage(stage, release)
    return result


def coverage_rows(frame, split, feature_set):
    return [[feature_set, split, int(row["ScheduledDepHour"]), int(row["Flights"]),
             int(row["Covered"]), float(row["Covered"] / row["Flights"])]
            for row in (frame.groupBy("ScheduledDepHour")
                        .agg(F.count("*").alias("Flights"),
                             F.sum((F.col("OpsOriginDepartedCount3h") >= 5).cast("long")).alias("Covered"))
                        .orderBy("ScheduledDepHour").collect())]


def bucket_rows(train, feature_set):
    dep_bucket = (F.when(F.col("OpsOriginMeanDepDelay3h") <= 0, "<=0")
                  .when(F.col("OpsOriginMeanDepDelay3h") <= 15, "1-15")
                  .when(F.col("OpsOriginMeanDepDelay3h") <= 60, "16-60").otherwise(">60"))
    backlog_bucket = (F.when(F.col("OpsOriginBacklogShare3h") <= 0, "0")
                      .when(F.col("OpsOriginBacklogShare3h") <= 0.10, "(0,0.10]")
                      .when(F.col("OpsOriginBacklogShare3h") <= 0.25, "(0.10,0.25]")
                      .when(F.col("OpsOriginBacklogShare3h") <= 0.50, "(0.25,0.50]").otherwise(">0.50"))
    rows = []
    for feature, expression in (("OpsOriginMeanDepDelay3h", dep_bucket),
                                ("OpsOriginBacklogShare3h", backlog_bucket)):
        for row in (train.withColumn("Bucket", expression).groupBy("Bucket")
                    .agg(F.count("*").alias("n"), F.sum("label").alias("p"), F.avg("label").alias("prev"))
                    .orderBy("Bucket").collect()):
            rows.append([feature_set, "train", feature, row["Bucket"], int(row["n"]),
                         int(row["p"]), float(row["prev"])])
    return rows


def univariate_rows(validation, feature_set, ml):
    total = validation.count()
    positives = int(validation.agg(F.sum("label")).first()[0])
    rows = []
    for feature in OPS_FEATURES:
        scored = validation.select("label", F.col(feature).cast("double").alias("score"))
        roc, pr = auc_pair(scored)
        top = ml.ranking_metrics(scored, [0.10], total, positives)[0]
        rows.append([feature_set, "validation", feature, roc, pr, top[3], top[4], top[5]])
    return rows


def saved_schedule_baseline():
    with open(os.path.join(FINAL_LOCAL, "validation_metrics.csv"), newline="", encoding="utf-8") as handle:
        metric = next(row for row in csv.DictReader(handle) if row["Model"] == "LogisticRegression")
    diagnostic_path = "/results/experiments/ml_redesign/diagnostics/gains_lift.csv"
    with open(diagnostic_path, newline="", encoding="utf-8") as handle:
        gain = next(row for row in csv.DictReader(handle)
                    if row["Split"] == "validation" and row["Model"] == "LogisticRegression" and
                    abs(float(row["TargetFlaggedFraction"]) - 0.10) < 1e-12)
    return {
        "roc": float(metric["AUC_ROC"]), "pr": float(metric["AUC_PR"]),
        "top10_precision": float(gain["Precision"]), "top10_recall": float(gain["Recall"]),
        "top10_lift": float(gain["Lift"]), "rows": 173549, "positives": 29526, "features": 56,
        "source": "frozen actual_final validation_metrics.csv plus diagnostics validation gains_lift.csv",
    }


def main():
    args = parse_args()
    lead = args.lead_minutes
    feature_set = f"operations-aware (T-{lead})"
    ml = load_ml()
    with open(os.path.join(FINAL_LOCAL, "frozen_config.json"), encoding="utf-8") as handle:
        frozen = json.load(handle)
    spec = dict(frozen["feature_spec"])
    spark = (SparkSession.builder.appName(f"BDA-AsOf-Ops-Phase1-T{lead}")
             .config("spark.sql.adaptive.enabled", "false")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.default.parallelism", "16")
             .config("spark.ui.retainedJobs", "100")
             .config("spark.ui.retainedStages", "100")
             .config("spark.sql.ui.retainedExecutions", "50").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    frames = []

    def keep(frame):
        cached = frame.persist(StorageLevel.MEMORY_AND_DISK)
        frames.append(cached)
        return cached

    def release(frame):
        if frame is not None:
            frame.unpersist()
        frames[:] = [cached for cached in frames if cached is not frame]

    try:
        prepared_path = f"{OUT_HDFS}/prepared/lead_{lead}"
        if args.mode == "model":
            with_ops = keep(spark.read.parquet(prepared_path))
            train = keep(with_ops.filter(F.col("FlightDate") < F.lit(VALIDATION_START)))
            validation = keep(with_ops.filter(F.col("FlightDate") >= F.lit(VALIDATION_START)))
            print(f"PREPARED_ROWS={with_ops.count()} TRAIN_ROWS={train.count()} VALIDATION_ROWS={validation.count()}", flush=True)
            original_base = list(ml.BASE_NUMERIC_FEATURES)
            ml.BASE_NUMERIC_FEATURES[:] = original_base + OPS_FEATURES
            try:
                operations = score_model(ml, train, validation, spec, keep, release)
            finally:
                ml.BASE_NUMERIC_FEATURES[:] = original_base
            baseline = saved_schedule_baseline()
            metric_rows = [
                ["schedule-only", lead, baseline["rows"], baseline["positives"], baseline["features"],
                 baseline["roc"], baseline["pr"], baseline["top10_precision"], baseline["top10_recall"],
                 baseline["top10_lift"], baseline["source"]],
                [feature_set, lead, operations["rows"], operations["positives"], operations["features"],
                 operations["roc"], operations["pr"], operations["top10_precision"],
                 operations["top10_recall"], operations["top10_lift"], "fresh Phase-1 YARN fit"],
            ]
            write_csv(spark, f"quick_lr_validation_lead{lead}.csv",
                      ["FeatureSet", "LeadMinutes", "ValidationRows", "Delayed15Plus", "FeatureCount", "AUC_ROC",
                       "AUC_PR", "Top10Precision", "Top10Recall", "Top10Lift", "MeasurementSource"], metric_rows, lead)
            if lead == 120:
                gain = operations["pr"] - GATE_REFERENCE_PR
                decision = "PASS" if gain >= GATE_GAIN else "STOP"
                write_csv(spark, "phase1_gate.csv",
                          ["ComparisonBaseline", "Candidate", "ReferenceValidationAUC_PR", "CandidateValidationAUC_PR",
                           "Gain", "RequiredGain", "GateDecision"],
                          [["schedule-only", "operations-aware (T-120)", GATE_REFERENCE_PR,
                            operations["pr"], gain, GATE_GAIN, decision]], lead)
                print(f"PHASE1_GATE={decision} gain={gain:.8f} candidate_pr={operations['pr']:.8f}", flush=True)
            print("ASOF_OPS_PHASE1_STATUS=SUCCEEDED", flush=True)
            return 0

        source_columns = sorted(set(ml.REQUIRED_CLEAN_COLUMNS) |
                                    {"DepDelay", "ArrDelay", "CancellationCode"})
        flights = keep(spark.read.parquet(CLEAN).select(*source_columns)
                       .filter(F.col("FlightDate") < F.lit(TEST_START)))
        print(f"SELECTION_SOURCE_ROWS={flights.count()}", flush=True)
        holidays = spark.read.option("header", True).schema("FlightDate date, HolidayPeriod string").csv(HOLIDAYS)
        demand = keep(ml.compute_demand_table(flights))
        eligible = keep(ml.build_eligible_frame(flights, holidays, demand))
        eligible_n = eligible.count()
        print(f"ELIGIBLE_TRAIN_VALIDATION_ROWS={eligible_n}", flush=True)

        events = keep(add_time_columns(flights, ml).select(
            "FlightDate", "Reporting_Airline", "Origin", "Dest", "Cancelled", "Diverted",
            "DepDelay", "ArrDelay", "ArrDel15", "_sched_dep_abs_hour",
            "_actual_dep_abs_hour", "_actual_arr_abs_hour"))
        events.count()
        priors = aggregate_priors(events)
        ops_keys, priors = build_ops_features(events, eligible, lead, priors, keep, release)
        ops_keys = keep(ops_keys)
        ops_keys.count()

        expected = eligible.count()
        with_ops = keep(eligible.join(ops_keys, JOIN_KEYS, "left"))
        check_count("eligible operations feature join", expected, with_ops)
        missing = with_ops.agg(*[F.sum(F.col(c).isNull().cast("long")).alias(c) for c in OPS_FEATURES]).first().asDict()
        if any(int(value or 0) for value in missing.values()):
            raise RuntimeError(f"operations features contain nulls: {missing}")

        (with_ops.repartition(16).write.mode("overwrite").parquet(prepared_path))
        print(f"WROTE_PREPARED={prepared_path}", flush=True)

        train = keep(with_ops.filter(F.col("FlightDate") < F.lit(VALIDATION_START)))
        validation = keep(with_ops.filter(F.col("FlightDate") >= F.lit(VALIDATION_START)))
        train_n, val_n = train.count(), validation.count()
        print(f"TRAIN_ROWS={train_n} VALIDATION_ROWS={val_n}", flush=True)

        write_csv(spark, f"coverage_by_hour_lead{lead}.csv",
                  ["FeatureSet", "Split", "ScheduledDepartureHour", "Flights", "FlightsWithAtLeast5OriginEvents", "CoverageShare"],
                  coverage_rows(train, "train", feature_set) + coverage_rows(validation, "validation", feature_set), lead)
        write_csv(spark, f"train_bucket_prevalence_lead{lead}.csv",
                  ["FeatureSet", "Split", "Feature", "Bucket", "Flights", "Delayed15Plus", "Prevalence"],
                  bucket_rows(train, feature_set), lead)
        write_csv(spark, f"univariate_validation_lead{lead}.csv",
                  ["FeatureSet", "Split", "Feature", "AUC_ROC", "AUC_PR", "Top10Precision", "Top10Recall", "Top10Lift"],
                  univariate_rows(validation, feature_set, ml), lead)

        if args.mode == "prepare":
            metadata = {
                "lead_minutes": lead, "feature_set": feature_set, "test_window_read": False,
                "train_end": "2024-11-24", "validation": ["2024-11-25", "2024-12-03"],
                "event_hour_rule": "Only event hours strictly before the prediction-time hour are included.",
                "arrival_date_approximation": "Scheduled arrival is FlightDate plus one day when CRSArrTime < CRSDepTime; ArrDelay is then added.",
                "smoothing_priors_train_only": priors, "smoothing_pseudo_events": SMOOTHING,
                "features": OPS_FEATURES, "lr_reg_param": 0.001, "schedule_spec": spec,
            }
            os.makedirs(OUT_LOCAL, exist_ok=True)
            with open(os.path.join(OUT_LOCAL, f"phase1_metadata_lead{lead}.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
                handle.write("\n")
            print("ASOF_OPS_PHASE1_STATUS=SUCCEEDED", flush=True)
            return 0

        metric_rows = []
        baseline = saved_schedule_baseline()
        metric_rows.append(["schedule-only", lead, baseline["rows"], baseline["positives"], baseline["features"],
                            baseline["roc"], baseline["pr"], baseline["top10_precision"],
                            baseline["top10_recall"], baseline["top10_lift"], baseline["source"]])

        original_base = list(ml.BASE_NUMERIC_FEATURES)
        ml.BASE_NUMERIC_FEATURES[:] = original_base + OPS_FEATURES
        try:
            operations = score_model(ml, train, validation, spec, keep, release)
        finally:
            ml.BASE_NUMERIC_FEATURES[:] = original_base
        metric_rows.append([feature_set, lead, operations["rows"], operations["positives"], operations["features"],
                            operations["roc"], operations["pr"], operations["top10_precision"],
                            operations["top10_recall"], operations["top10_lift"], "fresh Phase-1 YARN fit"])
        write_csv(spark, f"quick_lr_validation_lead{lead}.csv",
                  ["FeatureSet", "LeadMinutes", "ValidationRows", "Delayed15Plus", "FeatureCount", "AUC_ROC",
                   "AUC_PR", "Top10Precision", "Top10Recall", "Top10Lift", "MeasurementSource"], metric_rows, lead)

        gate_rows = []
        if lead == 120:
            gain = operations["pr"] - GATE_REFERENCE_PR
            decision = "PASS" if gain >= GATE_GAIN else "STOP"
            gate_rows.append(["schedule-only", "operations-aware (T-120)", GATE_REFERENCE_PR,
                              operations["pr"], gain, GATE_GAIN, decision])
            write_csv(spark, "phase1_gate.csv",
                      ["ComparisonBaseline", "Candidate", "ReferenceValidationAUC_PR", "CandidateValidationAUC_PR",
                       "Gain", "RequiredGain", "GateDecision"], gate_rows, lead)
            print(f"PHASE1_GATE={decision} gain={gain:.8f} candidate_pr={operations['pr']:.8f}", flush=True)

        metadata = {
            "lead_minutes": lead, "feature_set": feature_set, "test_window_read": False,
            "train_end": "2024-11-24", "validation": ["2024-11-25", "2024-12-03"],
            "event_hour_rule": "Only event hours strictly before the prediction-time hour are included.",
            "arrival_date_approximation": "Scheduled arrival is FlightDate plus one day when CRSArrTime < CRSDepTime; ArrDelay is then added.",
            "smoothing_priors_train_only": priors, "smoothing_pseudo_events": SMOOTHING,
            "features": OPS_FEATURES, "lr_reg_param": 0.001, "schedule_spec": spec,
        }
        os.makedirs(OUT_LOCAL, exist_ok=True)
        with open(os.path.join(OUT_LOCAL, f"phase1_metadata_lead{lead}.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        print("ASOF_OPS_PHASE1_STATUS=SUCCEEDED", flush=True)
        return 0
    except Exception as exc:
        print(f"ASOF_OPS_PHASE1_STATUS=FAILED: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        for frame in reversed(frames):
            try:
                frame.unpersist()
            except Exception:
                pass
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
