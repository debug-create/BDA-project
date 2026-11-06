#!/usr/bin/env python3
"""Leakage-safe Spark export for host boosted and deep models.

The default ``pretest`` scope physically filters the source before outcomes are
used and writes only train/validation.  ``test`` is intentionally a separate,
post-freeze action and must never be invoked by model-selection workflows.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

from pyspark import StorageLevel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window, functions as F

if "/src" not in sys.path:
    sys.path.insert(0, "/src")
import asof_ops_utc as utc
import asof_ops_state as state

spec = importlib.util.spec_from_file_location("ml_stage", "/src/03_ml.py")
ml = importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)

CLEAN = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
LOCAL_ROOT = "/results/experiments/ml_redesign/ops_v2/host_export"
HDFS_ROOT = "hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/host_export"
FROZEN_CONFIG = "/results/experiments/ml_redesign/ops_v2/frozen_ops_v2_config.json"
VAL_START = "2024-11-25"
TEST_START = "2024-12-04"
OPS_PRIORS = {
    "arr_mean": -0.9449691734090566, "arr_share": 0.13742487686689825,
    "carrier_backlog_share": 0.021607663028443753,
    "dep_mean": 5.679958829198303, "dep_share": 0.1384532103578844,
    "origin_backlog_share": 0.05213970052513455, "smoothing": 20.0,
}
NATURAL_KEY = ["FlightDate", "Origin", "Dest", "Reporting_Airline", "CRSDepTime"]


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pretest", "test"], default="pretest")
    return parser.parse_args()


def export_base(source, holidays):
    """Existing schedule features plus raw labels/keys retained for export."""
    raw = source.filter(ml.eligible_condition())
    # Duplicate natural keys exist because the source omits flight number/tail.
    # The technical ordinal may use schedule fields only: outcomes must not
    # influence even metadata identity at prediction time.
    order = [F.col(name).asc_nulls_last() for name in
             ["CRSArrTime", "CRSElapsedTime", "Distance", "DepTimeBlk"]]
    ordinal = F.row_number().over(Window.partitionBy(*NATURAL_KEY).orderBy(*order))
    raw = raw.withColumn("FlightKeyOrdinal", ordinal)
    raw = raw.withColumn("FlightKey", F.sha2(F.concat_ws("|", *[
        F.coalesce(F.col(column).cast("string"), F.lit("<NULL>")) for column in NATURAL_KEY
    ], F.col("FlightKeyOrdinal").cast("string")), 256))

    demand = ml.compute_demand_table(source)
    dest_demand = ml.compute_dest_demand_table(source)
    origin_day = demand.groupBy("FlightDate", "Origin").agg(F.sum("_sched_n").alias("_origin_day_n"))
    prev_hour = (demand.filter(F.col("_dep_hour_key") >= 0)
                 .select("FlightDate", "Origin", (F.col("_dep_hour_key") + 1).alias("_dep_hour_key"),
                         F.col("_sched_n").alias("_prev_hour_n")))
    two_pi = 2.0 * math.pi / 1440.0
    out = (raw.withColumn("_dep_min", ml.minute_of_day(F.col("CRSDepTime")))
           .withColumn("_arr_min", ml.minute_of_day(F.col("CRSArrTime")))
           .withColumn("_dep_hour_key", F.coalesce(F.floor(F.col("_dep_min") / 60).cast("int"), F.lit(-1)))
           .withColumn("_arr_hour_key", F.coalesce(F.floor(F.col("_arr_min") / 60).cast("int"), F.lit(-1)))
           .join(F.broadcast(holidays), "FlightDate", "left")
           .join(demand, ["FlightDate", "Origin", "_dep_hour_key"], "left")
           .join(prev_hour, ["FlightDate", "Origin", "_dep_hour_key"], "left")
           .join(origin_day, ["FlightDate", "Origin"], "left")
           .join(dest_demand, ["FlightDate", "Dest", "_arr_hour_key"], "left")
           .withColumn("HolidayPeriod", F.coalesce("HolidayPeriod", F.lit("Regular day")))
           .withColumn("label", F.col("ArrDel15").cast("double"))
           .withColumn("Reporting_Airline", F.coalesce("Reporting_Airline", F.lit("UNKNOWN")))
           .withColumn("DepTimeBlk", F.coalesce("DepTimeBlk", F.lit("UNKNOWN")))
           .withColumn("DayOfWeekCat", F.coalesce(F.concat(F.lit("DOW_"), F.col("DayOfWeek")), F.lit("DOW_UNKNOWN")))
           .withColumn("TimeParseInvalid", (F.col("_dep_min").isNull() | F.col("_arr_min").isNull()).cast("int"))
           .withColumn("DepMinuteSin", F.coalesce(F.sin(F.lit(two_pi) * F.col("_dep_min")), F.lit(0.0)))
           .withColumn("DepMinuteCos", F.coalesce(F.cos(F.lit(two_pi) * F.col("_dep_min")), F.lit(0.0)))
           .withColumn("ArrMinuteSin", F.coalesce(F.sin(F.lit(two_pi) * F.col("_arr_min")), F.lit(0.0)))
           .withColumn("ArrMinuteCos", F.coalesce(F.cos(F.lit(two_pi) * F.col("_arr_min")), F.lit(0.0)))
           .withColumn("ScheduledDepHour", F.coalesce(F.floor(F.col("_dep_min") / 60).cast("double"), F.lit(0.0)))
           .withColumn("ScheduledArrHour", F.coalesce(F.floor(F.col("_arr_min") / 60).cast("double"), F.lit(0.0)))
           .withColumn("IsHolidayWindow", (F.col("HolidayPeriod") != "Regular day").cast("double"))
           .withColumn("OriginHourDemandLog", F.log1p(F.coalesce("_sched_n", F.lit(0)).cast("double")))
           .withColumn("DestHourDemandLog", F.log1p(F.coalesce("_dest_sched_n", F.lit(0)).cast("double")))
           .withColumn("OriginDayDemandLog", F.log1p(F.coalesce("_origin_day_n", F.lit(0)).cast("double")))
           .withColumn("OriginPrevHourDemandLog", F.log1p(F.coalesce("_prev_hour_n", F.lit(0)).cast("double")))
           .withColumn("SchedSpeedMph", F.when(F.col("CRSElapsedTime") > 0,
                                                F.col("Distance") / (F.col("CRSElapsedTime") / 60.0)))
           .withColumn("DistanceLog", F.when(F.col("Distance") >= 0, F.log1p("Distance"))))
    return out


def add_history(train, validation, keep):
    specification = next(item for item in ml.EXPERIMENT_SPECS if item["name"] == "E13_e6_plus_asof")
    train_h, val_h = train, validation
    for group in ml.history_groups(specification):
        prior, tables = ml.build_history_tables(train, keep, group["window_days"], group["keys_map"])
        train_h = ml.apply_history_training_past_only(train_h, group["window_days"], group["keys_map"])
        val_h = ml.apply_history(val_h, prior, tables, group["keys_map"])
    return keep(train_h), keep(val_h), specification


def add_congestion(train, validation):
    medians = train.groupBy("Origin").agg(
        F.percentile_approx(F.coalesce("_sched_n", F.lit(0)).cast("double"), 0.5, 10000).alias("_origin_median_hour"))
    global_median = float(train.agg(
        F.percentile_approx(F.coalesce("_sched_n", F.lit(0)).cast("double"), 0.5, 10000)).first()[0] or 1.0)
    def apply(frame):
        return (frame.join(medians, "Origin", "left")
                .withColumn("OriginCongestionRatio",
                            F.coalesce("_sched_n", F.lit(0)).cast("double") /
                            F.greatest(F.coalesce("_origin_median_hour", F.lit(global_median)), F.lit(1.0)))
                .drop("_origin_median_hour"))
    return apply(train), apply(validation), global_median


def add_summary_features(frame):
    # One-hour subwindow ending at T uses the most recent complete UTC hour.
    mappings = {
        "OpsOriginDepartedCount1hLog": "seq_origin_departures_count_h1",
        "OpsOriginMeanDepDelay1h": "seq_origin_departures_mean_delay_h1",
        "OpsOriginDepDelay15Share1h": "seq_origin_departures_delay15_share_h1",
        "OpsOriginBacklogCount1hLog": "seq_origin_departures_backlog_count_h1",
        "OpsOriginInboundCount1hLog": "seq_origin_inbound_count_h1",
        "OpsOriginInboundMeanArrDelay1h": "seq_origin_inbound_mean_delay_h1",
        "OpsOriginInboundArrDelay15Share1h": "seq_origin_inbound_delay15_share_h1",
        "OpsCarrierDepartedCount1hLog": "seq_carrier_network_count_h1",
        "OpsCarrierMeanDepDelay1h": "seq_carrier_network_mean_delay_h1",
        "OpsCarrierDepDelay15Share1h": "seq_carrier_network_delay15_share_h1",
        "OpsDestLandedCount1hLog": "seq_destination_arrivals_count_h1",
        "OpsDestMeanArrDelay1h": "seq_destination_arrivals_mean_delay_h1",
        "OpsDestArrDelay15Share1h": "seq_destination_arrivals_delay15_share_h1",
    }
    out = frame
    for target, source in mappings.items():
        out = out.withColumn(target, F.col(source).cast("double"))
    for prefix, channel in [
        ("OriginDep", "origin_departures_delay15_share"),
        ("OriginInbound", "origin_inbound_delay15_share"),
        ("Carrier", "carrier_network_delay15_share"),
        ("DestArr", "destination_arrivals_delay15_share")]:
        prior2 = (F.col(f"seq_{channel}_h2") + F.col(f"seq_{channel}_h3")) / 2.0
        out = out.withColumn(f"Ops{prefix}DelayShareTrend1hVsPrior2h",
                             F.col(f"seq_{channel}_h1") - prior2)
    out = (out.withColumn("Route", F.concat_ws("-", "Origin", "Dest"))
           .withColumn("RouteHash65536", F.pmod(F.xxhash64("Origin", "Dest"), F.lit(65536)).cast("int"))
           .withColumn("Split", F.when(F.col("FlightDate") < F.lit(VAL_START), "train")
                       .when(F.col("FlightDate") < F.lit(TEST_START), "validation").otherwise("test"))
           .withColumn("FitEligible", (F.col("FlightDate") >= F.lit("2024-10-08")).cast("int"))
           .withColumn("arr_minutes", F.greatest(F.lit(-30.0), F.least(F.lit(180.0), F.col("ArrDelay"))).cast("float"))
           .withColumn("dep_del15", (F.col("DepDelay") >= 15).cast("int"))
           .withColumn("arr_bucket", F.when(F.col("ArrDelay") <= 0, 0).when(F.col("ArrDelay") < 15, 1)
                       .when(F.col("ArrDelay") < 30, 2).when(F.col("ArrDelay") < 60, 3).otherwise(4).cast("int")))
    return out


def add_ops_masks(frame):
    count_for = {}
    for name in ml.OPS_FEATURES:
        if name.startswith("OpsOriginInbound"):
            count = "OpsOriginInboundLandedCount3h"
        elif name.startswith("OpsOriginBacklog"):
            count = "OpsOriginBacklogCount3h"
        elif name.startswith("OpsOrigin"):
            count = "OpsOriginDepartedCount3h"
        elif name.startswith("OpsCarrier"):
            count = "OpsCarrierDepartedCountToday"
        else:
            count = "OpsDestLandedCount3h"
        count_for[name] = count
    out = frame
    for name, count in count_for.items():
        out = (out.withColumn(f"{name}__count", F.col(count).cast("float"))
               .withColumn(f"{name}__mask", (F.col(count) > 0).cast("int")))
    return out


def main():
    config = args(); started = time.time()
    spark = (SparkSession.builder.appName(f"OpsV2HostExport-{config.scope}")
             .config("spark.sql.shuffle.partitions", "32").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.setCheckpointDir(f"{HDFS_ROOT}/_checkpoints_{config.scope}")
    cached = []
    def keep(frame):
        frame.persist(StorageLevel.DISK_ONLY); cached.append(frame); return frame
    def release(frame):
        try: frame.unpersist()
        except Exception: pass
        cached[:] = [item for item in cached if item is not frame]
    try:
        clean = spark.read.parquet(CLEAN)
        if config.scope == "pretest":
            source = clean.filter(F.col("FlightDate") < F.lit(TEST_START))
        else:
            if not os.path.isfile(FROZEN_CONFIG):
                raise RuntimeError(f"post-freeze export refused: missing frozen config {FROZEN_CONFIG}")
            with open(FROZEN_CONFIG, encoding="utf-8") as frozen_file:
                frozen = json.load(frozen_file)
            if (not frozen.get("decisions_frozen") or not frozen.get("committed_before_test") or
                    not frozen.get("test_authorized") or
                    not (frozen.get("phase1_passed") or frozen.get("phase2_passed"))):
                raise RuntimeError("post-freeze export refused: frozen config does not authorize test access")
            print("DECISIONS_FROZEN=1: opening test source for post-freeze export", flush=True)
            source = clean
        holidays = (spark.read.option("header", True).csv(HOLIDAYS)
                    .select(F.to_date("FlightDate").alias("FlightDate"), "HolidayPeriod"))
        base = keep(export_base(source, holidays)); base_n = base.count()
        tz = keep(utc.load_timezone_offsets(spark)); tz.count()
        fitting_cutoff = TEST_START if config.scope == "test" else VAL_START
        with_ops, priors, coverage = utc.attach_asof_features_utc(
            source, base, 120, fitting_cutoff, tz, keep, release, priors=None)
        with_ops = keep(with_ops.checkpoint(eager=True))
        with_state, state_columns = state.attach_hourly_state_utc(
            source, with_ops, tz, 120, priors, keep, release)
        with_state = keep(with_state.checkpoint(eager=True))
        if config.scope == "pretest":
            fitting = with_state.filter(F.col("FlightDate") < F.lit(VAL_START))
            apply = with_state.filter((F.col("FlightDate") >= F.lit(VAL_START)) &
                                      (F.col("FlightDate") < F.lit(TEST_START)))
        else:
            fitting = with_state.filter(F.col("FlightDate") < F.lit(TEST_START))
            apply = with_state.filter(F.col("FlightDate") >= F.lit(TEST_START))
        fitting_h, apply_h, specification = add_history(fitting, apply, keep)
        fitting_h, apply_h, congestion_median = add_congestion(fitting_h, apply_h)
        assembled = keep(add_ops_masks(add_summary_features(fitting_h.unionByName(apply_h))))

        # Pretest: fit on train only.  Authorized test export: refit on the
        # complete train+validation fitting window, never on a test label.
        if config.scope == "pretest":
            fitting_rows = assembled.filter((F.col("Split") == "train") & (F.col("FitEligible") == 1))
        else:
            fitting_rows = assembled.filter((F.col("Split").isin("train", "validation")) &
                                             (F.col("FitEligible") == 1))
        pipeline = ml.make_pipeline(specification).fit(fitting_rows)
        fitting_vec = keep(pipeline.transform(fitting_rows).select("FlightKey", "label", "features"))
        lr = ml.fit_logistic(fitting_vec, 0.001)
        all_vec = pipeline.transform(assembled).select("FlightKey", "features")
        offsets = lr.transform(all_vec).select(
            "FlightKey", F.least(F.lit(1 - 1e-7), F.greatest(F.lit(1e-7),
             vector_to_array("probability")[1])).alias("_lr_p"))
        offsets = offsets.withColumn("lr_offset", F.log(F.col("_lr_p") / (1 - F.col("_lr_p")))).drop("_lr_p")
        exported = assembled.join(offsets, "FlightKey", "left")

        # Train-only median imputation for the raw dense columns used by host models.
        impute, plain, history, _ = ml.spec_columns(specification)
        median_fitting = exported.filter((F.col("Split") == "train") & (F.col("FitEligible") == 1)) if config.scope == "pretest" else exported.filter(
            (F.col("Split").isin("train", "validation")) & (F.col("FitEligible") == 1))
        median_row = median_fitting.agg(*[
            F.percentile_approx(F.col(column).cast("double"), 0.5, 10000).alias(column)
            for column in impute]).first().asDict()
        medians = {column: float(median_row[column] or 0.0) for column in impute}
        for column in impute:
            exported = exported.withColumn(column, F.coalesce(F.col(column).cast("double"), F.lit(medians[column])))

        extra_dense = [column for column in exported.columns if column.startswith("Ops") or
                       column.startswith("RouteEarlier") or column == "OriginCongestionRatio"]
        dense_columns = list(dict.fromkeys(impute + plain + history + extra_dense))
        category_columns = ["Reporting_Airline", "Origin", "Dest", "Route", "ScheduledDepHour",
                            "DayOfWeekCat", "HolidayPeriod", "DepTimeBlk", "RouteHash65536"]
        key_label_columns = NATURAL_KEY + ["FlightKeyOrdinal", "FlightKey", "Split", "FitEligible",
                                           "label", "arr_minutes", "dep_del15", "arr_bucket", "lr_offset"]
        output_columns = key_label_columns + category_columns + dense_columns + state_columns
        output_columns = list(dict.fromkeys(output_columns))
        # Truncate the offset/history/state lineage once.  All reconciliation
        # actions and both writes below must read this same immutable frame,
        # rather than independently rebuilding the full feature graph.
        exported = keep(exported.select(*output_columns).checkpoint(eager=True))
        exported.count()

        counts = {row["Split"]: int(row["count"]) for row in exported.groupBy("Split").count().collect()}
        prevalence = {row["Split"]: float(row["prevalence"]) for row in
                      exported.groupBy("Split").agg(F.avg("label").alias("prevalence")).collect()}
        natural_dup = exported.groupBy(*NATURAL_KEY).count().filter(F.col("count") > 1)
        natural_dup_groups = natural_dup.count()
        technical_dup = exported.groupBy("FlightKey").count().filter(F.col("count") > 1).count()
        null_row = exported.agg(*[F.sum(F.col(column).isNull().cast("long")).alias(column)
                                  for column in exported.columns]).first().asDict()
        nulls = {key: int(value or 0) for key, value in null_row.items()}
        non_label_nulls = {key: value for key, value in nulls.items() if value and key not in
                           {"arr_minutes", "dep_del15", "arr_bucket"}}
        if technical_dup or non_label_nulls:
            raise RuntimeError(f"export reconciliation failed technical_dup={technical_dup} nulls={non_label_nulls}")
        expected = {"train": 1067747, "validation": 173549}
        if config.scope == "pretest" and counts != expected:
            raise RuntimeError(f"row reconciliation failed {counts} != {expected}")

        out_local = f"{LOCAL_ROOT}/{config.scope}"
        out_hdfs = f"{HDFS_ROOT}/{config.scope}"
        # The cluster's Hadoop default filesystem is HDFS, so an absolute
        # /results path is otherwise resolved as HDFS too. Force the host
        # bind-mount write onto the container's local filesystem.
        local_uri = f"file://{out_local}"
        exported.repartition(32, "Split").write.mode("overwrite").partitionBy("Split").parquet(local_uri)
        exported.repartition(32, "Split").write.mode("overwrite").partitionBy("Split").parquet(out_hdfs)
        metadata = {
            "scope": config.scope, "test_opened": config.scope == "test", "rows": counts,
            "prevalence": prevalence, "natural_key_duplicate_groups": natural_dup_groups,
            "technical_flight_key_duplicates": technical_dup, "null_counts": nulls,
            "dense_columns": dense_columns, "state_columns": state_columns,
            "categorical_columns": category_columns, "imputation_medians_train_only": medians,
            "ops_priors": priors, "utc_coverage": coverage, "origin_congestion_global_median": congestion_median,
            "runtime_seconds": time.time() - started,
        }
        os.makedirs(LOCAL_ROOT, exist_ok=True)
        with open(f"{LOCAL_ROOT}/{config.scope}_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2); handle.write("\n")
        print("HOST_EXPORT_RECONCILIATION=" + json.dumps({
            "rows": counts, "prevalence": prevalence, "natural_key_duplicate_groups": natural_dup_groups,
            "technical_key_duplicates": technical_dup, "non_label_nulls": non_label_nulls}, sort_keys=True), flush=True)
        print("HOST_EXPORT_STATUS=SUCCEEDED", flush=True)
        return 0
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"HOST_EXPORT_STATUS=FAILED error={type(exc).__name__}: {exc}", flush=True)
        return 2
    finally:
        for frame in reversed(cached):
            try: frame.unpersist()
            except Exception: pass
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
