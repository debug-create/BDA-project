#!/usr/bin/env python3
"""Read-only diagnostics for the frozen flight-delay models.

Large computations stay in Spark. Test-derived outputs are explicitly labelled
descriptive and are never used to choose a model, feature, threshold, or setting.
"""

import csv
import importlib.util
import json
import math
import os
import sys
from datetime import date

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.sql import SparkSession, Window, functions as F

ROOT = "/src"
FINAL_LOCAL = "/results/experiments/ml_redesign/actual_final"
OUT_LOCAL = "/results/experiments/ml_redesign/diagnostics"
OUT_HDFS = "hdfs://master:9000/user/root/flights/experiments/ml_redesign/diagnostics"
MODELS = "hdfs://master:9000/user/root/flights/experiments/ml_redesign/actual_final/models"
VALIDATION_STATUS = "VALIDATION - used for diagnostics/decisions"
TRAIN_STATUS = "TRAIN - diagnostic fit assessment"
TEST_STATUS = "TEST - DESCRIPTIVE ONLY, not used for any decision"
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def load_ml():
    spec = importlib.util.spec_from_file_location("flight_ml", os.path.join(ROOT, "03_ml.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(spark, name, header, rows):
    os.makedirs(OUT_LOCAL, exist_ok=True)
    path = os.path.join(OUT_LOCAL, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([["" if v is None else v for v in row] for row in rows])
    schema = ",".join(f"`{h}` string" for h in header)
    strings = [["" if v is None else str(v) for v in row] for row in rows]
    spark.createDataFrame(strings, schema).coalesce(1).write.mode("overwrite").option("header", True).csv(
        f"{OUT_HDFS}/{name[:-4] if name.endswith('.csv') else name}"
    )
    print(f"WROTE {path}", flush=True)


def keep(frames, df):
    cached = df.persist(StorageLevel.MEMORY_AND_DISK)
    frames.append(cached)
    return cached


def auc_pair(df):
    return histogram_auc_pair(df)


def histogram_auc_pair(df, bins=1000):
    """Bounded-memory AUC approximation from at most ``bins + 1`` aggregate rows."""
    limits = df.agg(F.min("score").alias("lo"), F.max("score").alias("hi")).first()
    lo, hi = float(limits["lo"]), float(limits["hi"])
    if hi <= lo:
        return 0.5, float(df.agg(F.avg("label")).first()[0])
    scaled = F.floor((F.col("score") - F.lit(lo)) / F.lit(hi - lo) * F.lit(bins)).cast("int")
    rows = (df.withColumn("_score_bin", scaled)
            .groupBy("_score_bin").agg(F.count("*").alias("n"), F.sum("label").alias("p"))
            .orderBy(F.desc("_score_bin")).collect())
    total_p = sum(int(r["p"] or 0) for r in rows)
    total_n = sum(int(r["n"]) - int(r["p"] or 0) for r in rows)
    tp = fp = 0
    previous_recall, previous_precision = 0.0, 1.0
    previous_fpr, previous_tpr = 0.0, 0.0
    pr_area = roc_area = 0.0
    for r in rows:
        positives = int(r["p"] or 0)
        negatives = int(r["n"]) - positives
        tp += positives
        fp += negatives
        recall = tp / total_p
        precision = tp / (tp + fp)
        fpr = fp / total_n
        tpr = recall
        pr_area += (recall - previous_recall) * (precision + previous_precision) / 2.0
        roc_area += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        previous_recall, previous_precision = recall, precision
        previous_fpr, previous_tpr = fpr, tpr
    return roc_area, pr_area


def score(model, frame, extra_cols):
    return model.transform(frame).select(
        *extra_cols, "label", vector_to_array(F.col("probability"))[1].alias("score")
    ).repartition(16)


def build_history_frames(ml, fit_base, apply_bases, frames):
    groups = ml.history_groups(SPEC)
    fit_h = fit_base
    applied = dict(apply_bases)
    group_state = []
    for group in groups:
        prior, tables = ml.build_history_tables(
            fit_base, lambda d: keep(frames, d), group["window_days"], group["keys_map"]
        )
        fit_h = ml.apply_history_training_past_only(fit_h, group["window_days"], group["keys_map"])
        applied = {name: ml.apply_history(df, prior, tables, group["keys_map"]) for name, df in applied.items()}
        group_state.append((group, prior, tables))
    fit_h = keep(frames, fit_h)
    applied = {name: keep(frames, df) for name, df in applied.items()}
    first = fit_base.agg(F.min("FlightDate")).first()[0]
    fit_h = keep(frames, fit_h.filter(F.col("FlightDate") >= F.date_add(F.lit(first), int(SPEC["warmup_days"]))))
    return fit_h, applied, group_state


def thin_history_rows(ml, fit_base, model_train, spark):
    rows = []
    total = model_train.count()
    for group in ml.history_groups(SPEC):
        for name, keys in group["keys_map"].items():
            daily = (fit_base.groupBy(*(keys + ["FlightDate"]))
                     .agg(F.count(F.lit(1)).alias("_day_n"))
                     .withColumn("_day", F.datediff("FlightDate", F.lit("1970-01-01").cast("date"))))
            base_w = Window.partitionBy(*keys).orderBy("_day")
            window = (base_w.rowsBetween(Window.unboundedPreceding, -1) if group["window_days"] is None
                      else base_w.rangeBetween(-int(group["window_days"]), -1))
            counts = daily.withColumn("_previous_n", F.coalesce(F.sum("_day_n").over(window), F.lit(0))).select(
                *(keys + ["FlightDate", "_previous_n"])
            )
            joined = model_train.select(*(keys + ["FlightDate"])).join(counts, keys + ["FlightDate"], "left")
            thin = joined.agg(F.sum(F.when(F.coalesce("_previous_n", F.lit(0)) < 100, 1).otherwise(0))).first()[0]
            rows.append([name, int(total), int(thin or 0), float(thin or 0) / total])
    write_csv(spark, "history_thin_share.csv", ["Feature", "TrainingRows", "RowsWithFewerThan100EarlierFlights", "Share"], rows)


def history_stats(ml, split_frames, spark):
    features = [name for group in ml.history_groups(SPEC) for name in group["keys_map"]]
    rows = []
    statuses = {"train": TRAIN_STATUS, "validation": VALIDATION_STATUS, "test": TEST_STATUS}
    for split, df in split_frames.items():
        exprs = []
        for c in features:
            exprs.extend([
                F.avg(c).alias(f"{c}__mean"), F.stddev_samp(c).alias(f"{c}__std"),
                F.percentile_approx(c, F.array(F.lit(0.05), F.lit(0.50), F.lit(0.95)), 10000).alias(f"{c}__q")
            ])
        result = df.agg(*exprs).first().asDict()
        for c in features:
            q = result[f"{c}__q"]
            rows.append([statuses[split], split, c, result[f"{c}__mean"], result[f"{c}__std"], q[0], q[1], q[2]])
    write_csv(spark, "history_feature_skew.csv",
              ["Status", "Split", "Feature", "Mean", "StdDev", "P05", "P50", "P95"], rows)


def calibration(scored, split, status, spark):
    # Avoid an executor-heavy global row-number sort on the constrained cluster.
    cuts = scored.approxQuantile("score", [i / 10.0 for i in range(1, 10)], 0.001)
    bin_expr = F.lit(10)
    for index, cut in reversed(list(enumerate(cuts, start=1))):
        bin_expr = F.when(F.col("score") <= F.lit(cut), F.lit(index)).otherwise(bin_expr)
    rows = (scored.withColumn("CalibrationBin", bin_expr)
            .groupBy("CalibrationBin")
            .agg(F.count("*").alias("n"), F.avg("score").alias("mean_score"), F.avg("label").alias("observed"))
            .orderBy("CalibrationBin").collect())
    return [[status, split, int(r["CalibrationBin"]), int(r["n"]), float(r["mean_score"]), float(r["observed"])] for r in rows]


def gains(ml, scored, model, split, status):
    agg = scored.agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
    values = ml.ranking_metrics(scored, FRACTIONS, int(agg["n"]), int(agg["p"]))
    return [[status, split, model, *row] for row in values]


def staleness(scored, split, model, threshold, start_date, status, unit):
    if unit == "day":
        grouped = scored.withColumn("Period", F.datediff("FlightDate", F.lit(start_date).cast("date")))
    else:
        grouped = scored.withColumn("Period", F.floor(F.datediff("FlightDate", F.lit(start_date).cast("date")) / 7) + 1)
    periods = [int(r[0]) for r in grouped.select("Period").distinct().orderBy("Period").collect()]
    rows = []
    for period in periods:
        part = grouped.filter(F.col("Period") == period)
        roc, pr = auc_pair(part)
        a = part.agg(F.count("*").alias("n"), F.avg("label").alias("prev"),
                     F.avg((F.col("score") >= threshold).cast("double")).alias("rate")).first()
        rows.append([status, split, model, unit, period, int(a["n"]), float(a["prev"]), roc, pr, float(a["rate"])])
    return rows


def single_feature_power(history_frame, ml, spark):
    features = [name for group in ml.history_groups(SPEC) for name in group["keys_map"]] + ["ScheduledDepHour"]
    rows = []
    for name in features:
        candidate = history_frame.select("label", F.col(name).cast("double").alias("score"))
        roc, pr = auc_pair(candidate)
        rows.append([VALIDATION_STATUS, name, roc, pr])
    write_csv(spark, "single_feature_ranking_validation.csv", ["Status", "FeatureUsedAlone", "AUC_ROC", "AUC_PR"], rows)


def histogram_ap(rows):
    ordered = sorted(rows, key=lambda x: x[0], reverse=True)
    positives = sum(r[2] for r in ordered)
    if positives == 0:
        return None
    cum_n = 0
    cum_p = 0
    ap = 0.0
    for _, n, p in ordered:
        cum_n += n
        cum_p += p
        ap += (p / positives) * (cum_p / cum_n)
    return ap


def group_signal(scored, group_col, group_type, allowed=None):
    base = scored if allowed is None else scored.filter(F.col(group_col).isin(allowed))
    groups = base.groupBy(group_col).agg(F.count("*").alias("n"), F.sum("label").alias("p")).collect()
    rows = []
    for group_row in groups:
        group_value = group_row[group_col]
        n, p = int(group_row["n"]), int(group_row["p"] or 0)
        if n < 2000 or p == 0 or p == n:
            continue
        part = base.filter(F.col(group_col) == F.lit(group_value))
        _, pr = auc_pair(part)
        prevalence = p / n
        cutoff = float(part.approxQuantile("score", [0.90], 0.001)[0])
        top = part.filter(F.col("score") >= cutoff).agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
        seen, pos = int(top["n"]), int(top["p"] or 0)
        precision = pos / seen if seen else 0.0
        rows.append([VALIDATION_STATUS, group_type, str(group_value), n, p, prevalence, pr, precision,
                     precision / prevalence if prevalence else None])
    return rows


def coefficients(lr_model, train_vec, names, spark):
    summary = train_vec.select(Summarizer.metrics("std").summary(F.col("features")).alias("s")).first()["s"]
    std = summary["std"].toArray().tolist()
    coefs = lr_model.coefficients.toArray().tolist()
    values = [(names[i], coefs[i], std[i], coefs[i] * std[i]) for i in range(len(names))]
    positive = sorted(values, key=lambda x: x[3], reverse=True)[:20]
    negative = sorted(values, key=lambda x: x[3])[:20]
    rows = [[VALIDATION_STATUS, "positive", i + 1, *v] for i, v in enumerate(positive)]
    rows += [[VALIDATION_STATUS, "negative", i + 1, *v] for i, v in enumerate(negative)]
    write_csv(spark, "lr_standardized_coefficients_top20.csv",
              ["Status", "Direction", "Rank", "Feature", "RawCoefficient", "TrainingStdDev", "StandardizedCoefficient"], rows)


def operating_thresholds(val, test, model, spark):
    # Validation-only threshold candidates from a fine score grid.
    hist = (val.withColumn("threshold", F.round("score", 4))
            .groupBy("threshold").agg(F.count("*").alias("n"), F.sum("label").alias("p"))
            .orderBy(F.desc("threshold")).collect())
    total_pos = int(val.agg(F.sum("label")).first()[0])
    cumulative = []
    n = p = 0
    for r in hist:
        n += int(r["n"])
        p += int(r["p"] or 0)
        cumulative.append((float(r["threshold"]), n, p, p / n, p / total_pos))
    choices = []
    for target in (0.30, 0.40, 0.50):
        eligible = [x for x in cumulative if x[3] >= target]
        choices.append((f"validation_precision_at_least_{int(target*100)}pct", max(eligible, key=lambda x: x[4])[0] if eligible else None))
    for fraction in (0.10, 0.20):
        choices.append((f"validation_top_{int(fraction*100)}pct", float(val.approxQuantile("score", [1 - fraction], 0.0001)[0])))
    rows = []
    for label, threshold in choices:
        if threshold is None:
            rows.append([VALIDATION_STATUS, model, label, "UNREACHABLE", None, None, None, None])
            rows.append([TEST_STATUS, model, label, "UNREACHABLE", None, None, None, None])
            continue
        for split, status, df in (("validation", VALIDATION_STATUS, val), ("test", TEST_STATUS, test)):
            a = df.agg(F.count("*").alias("total"), F.sum("label").alias("all_pos"),
                       F.sum(F.when(F.col("score") >= threshold, 1).otherwise(0)).alias("flag"),
                       F.sum(F.when((F.col("score") >= threshold) & (F.col("label") == 1), 1).otherwise(0)).alias("tp")).first()
            flag, tp = int(a["flag"]), int(a["tp"] or 0)
            rows.append([status, model, label, threshold, flag / int(a["total"]),
                         tp / flag if flag else 0.0, tp / int(a["all_pos"]), split])
    return rows


def save_score_histograms(scored_map):
    frames = []
    for (split, model), df in scored_map.items():
        status = TEST_STATUS if split == "test" else VALIDATION_STATUS
        frames.append(df.select(F.lit(status).alias("Status"), F.lit(split).alias("Split"),
                                F.lit(model).alias("Model"), "FlightDate",
                                F.floor(F.col("score") * 1000).cast("int").alias("ScoreBin"), "label"))
    unioned = frames[0]
    for frame in frames[1:]:
        unioned = unioned.unionByName(frame)
    (unioned.groupBy("Status", "Split", "Model", "FlightDate", "ScoreBin")
     .agg(F.count("*").alias("Flights"), F.sum("label").cast("long").alias("Delayed15Plus"))
     .coalesce(1).write.mode("overwrite").option("header", True).csv(f"file://{OUT_LOCAL}/score_histograms_1000bin"))
    (unioned.groupBy("Status", "Split", "Model", "FlightDate", "ScoreBin")
     .agg(F.count("*").alias("Flights"), F.sum("label").cast("long").alias("Delayed15Plus"))
     .write.mode("overwrite").option("header", True).csv(f"{OUT_HDFS}/score_histograms_1000bin"))


def main():
    global SPEC
    ml = load_ml()
    os.makedirs(OUT_LOCAL, exist_ok=True)
    with open(os.path.join(FINAL_LOCAL, "frozen_config.json"), encoding="utf-8") as handle:
        frozen = json.load(handle)
    with open(os.path.join(FINAL_LOCAL, "preprocessing_metadata.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    SPEC = frozen["feature_spec"]
    spark = (SparkSession.builder.appName("BDA-Flight-ML-Diagnostics")
             .config("spark.sql.adaptive.enabled", "false")
             .config("spark.sql.ui.explainMode", "simple")
             .config("spark.default.parallelism", "16")
             .config("spark.sql.shuffle.partitions", "16")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    frames = []
    try:
        flights = spark.read.parquet(ml.CLEAN_PATH)
        holidays = spark.read.option("header", True).option("inferSchema", True).csv(ml.HOLIDAYS_PATH).select(
            F.to_date("FlightDate").alias("FlightDate"), "HolidayPeriod")
        val_start = date.fromisoformat(frozen["windows"]["validation"][0])
        test_start = date.fromisoformat(frozen["windows"]["test"][0])
        pretest_source = flights.filter(F.col("FlightDate") < F.lit(test_start))
        pretest = keep(frames, ml.build_eligible_frame(pretest_source, holidays, ml.compute_demand_table(pretest_source)))
        train_base = keep(frames, pretest.filter(F.col("FlightDate") < F.lit(val_start)))
        val_base = keep(frames, pretest.filter(F.col("FlightDate") >= F.lit(val_start)))
        train_h, applied, group_state = build_history_frames(ml, train_base, {"validation": val_base}, frames)
        val_h = applied["validation"]
        train_h.count()
        val_h.count()
        for _, _, tables in group_state:
            for table in tables.values():
                table.unpersist()
        pipeline = ml.make_pipeline(SPEC).fit(train_h)
        extras = ["FlightDate", "Reporting_Airline", "Origin", "ScheduledDepHour", "DayOfWeekCat"]
        # Keep only the columns used below.  Caching the full transformed source row
        # exceeded the fixed 512 MiB executor envelope on real data.
        train_vec = keep(frames, pipeline.transform(train_h).select("label", "HolidayPeriod", "features"))
        val_vec = keep(frames, pipeline.transform(val_h).select(*extras, "label", "HolidayPeriod", "features"))
        train_vec.count()
        val_vec.count()
        lr = ml.fit_logistic(train_vec, frozen["logistic_regression"]["regParam"])
        rf = ml.fit_forest(train_vec, frozen["random_forest"])
        lr_train = keep(frames, score(lr, train_vec, []))
        lr_val = keep(frames, score(lr, val_vec, extras))
        rf_train = keep(frames, score(rf, train_vec, []))
        rf_val = keep(frames, score(rf, val_vec, extras))
        for cached_score in (lr_train, lr_val, rf_train, rf_val):
            cached_score.count()

        # Reconstruct the final serving path from saved tables/pipeline/models.
        test_source = flights.filter(F.col("FlightDate") >= F.lit(test_start))
        test_base = keep(frames, ml.build_eligible_frame(test_source, holidays, ml.compute_demand_table(test_source)))
        trainval = pretest
        test_h = test_base
        max_fit_date = trainval.agg(F.max("FlightDate")).first()[0]
        for group in ml.history_groups(SPEC):
            prior_base = trainval
            if group["window_days"] is not None:
                prior_base = trainval.filter(F.col("FlightDate") > F.date_sub(F.lit(max_fit_date), int(group["window_days"])))
            prior = float(prior_base.agg(F.avg("label")).first()[0])
            tables = {name: spark.read.parquet(f"{MODELS}/historical_rate_tables/{name}") for name in group["keys_map"]}
            test_h = ml.apply_history(test_h, prior, tables, group["keys_map"])
        test_h = keep(frames, test_h)
        test_h.count()
        saved_pipeline = PipelineModel.load(f"{MODELS}/feature_pipeline")
        test_vec = keep(frames, saved_pipeline.transform(test_h).select(*extras, "label", "HolidayPeriod", "features"))
        test_vec.count()
        saved_lr = LogisticRegressionModel.load(f"{MODELS}/logistic_regression")
        saved_rf = RandomForestClassificationModel.load(f"{MODELS}/random_forest")
        lr_test = keep(frames, score(saved_lr, test_vec, extras))
        rf_test = keep(frames, score(saved_rf, test_vec, extras))
        lr_test.count()
        rf_test.count()

        metric_rows = []
        for status, split, model, df in (
            (TRAIN_STATUS, "train", "LogisticRegression", lr_train),
            (VALIDATION_STATUS, "validation", "LogisticRegression", lr_val),
            (TRAIN_STATUS, "train", "RandomForest", rf_train),
            (VALIDATION_STATUS, "validation", "RandomForest", rf_val),
            (TEST_STATUS, "test", "LogisticRegression", lr_test),
            (TEST_STATUS, "test", "RandomForest", rf_test),
        ):
            roc, pr = auc_pair(df)
            a = df.agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
            method = "1000-bin histogram approximation (bounded-memory diagnostic)"
            metric_rows.append([status, split, model, int(a["n"]), int(a["p"]), int(a["p"]) / int(a["n"]), roc, pr, method])
        write_csv(spark, "train_validation_test_auc.csv",
                  ["Status", "Split", "Model", "Flights", "Delayed15Plus", "Prevalence", "AUC_ROC", "AUC_PR", "Method"], metric_rows)

        calibration_rows = calibration(lr_val, "validation", VALIDATION_STATUS, spark) + calibration(lr_test, "test", TEST_STATUS, spark)
        write_csv(spark, "lr_calibration_deciles.csv",
                  ["Status", "Split", "EqualCountBin", "Flights", "MeanPredictedProbability", "ObservedDelayRate"], calibration_rows)

        gain_rows = []
        for split, status, model, df in (
            ("validation", VALIDATION_STATUS, "LogisticRegression", lr_val),
            ("validation", VALIDATION_STATUS, "RandomForest", rf_val),
            ("test", TEST_STATUS, "LogisticRegression", lr_test),
            ("test", TEST_STATUS, "RandomForest", rf_test),
        ):
            gain_rows += gains(ml, df, model, split, status)
        write_csv(spark, "gains_lift.csv", ["Status", "Split", "Model", "TargetFlaggedFraction", "ActualFlaggedFraction",
                  "FlaggedFlights", "Precision", "Recall", "Lift"], gain_rows)

        history_stats(ml, {"train": train_h, "validation": val_h, "test": test_h}, spark)
        thin_history_rows(ml, train_base, train_h, spark)

        stale_rows = []
        for model, df, threshold in (("LogisticRegression", lr_val, 0.11), ("RandomForest", rf_val, 0.12)):
            stale_rows += staleness(df, "validation", model, threshold, "2024-11-24", VALIDATION_STATUS, "day")
        for model, df, threshold in (("LogisticRegression", lr_test, 0.11), ("RandomForest", rf_test, 0.12)):
            stale_rows += staleness(df, "test", model, threshold, "2024-12-04", TEST_STATUS, "week")
        write_csv(spark, "staleness_by_period.csv", ["Status", "Split", "Model", "PeriodUnit", "Period", "Flights",
                  "Prevalence", "AUC_ROC", "AUC_PR", "PredictedPositiveRate"], stale_rows)

        # Walk-forward validation: train-fitted pipeline/model, earlier validation labels only in refreshed history.
        rolling = trainval
        for group in ml.history_groups(SPEC):
            rolling = ml.apply_history_training_past_only(rolling, group["window_days"], group["keys_map"])
        rolling_val = keep(frames, rolling.filter(F.col("FlightDate") >= F.lit(val_start)))
        rolling_vec = keep(frames, pipeline.transform(rolling_val).select("FlightDate", "label", "HolidayPeriod", "features"))
        rolling_score = keep(frames, score(lr, rolling_vec, ["FlightDate"]))
        frozen_roc, frozen_pr = auc_pair(lr_val)
        rolling_roc, rolling_pr = auc_pair(rolling_score)
        write_csv(spark, "walk_forward_validation.csv", ["Status", "HistoryServingMode", "AUC_ROC", "AUC_PR", "DeltaAUC_PR_vsFrozen"], [
            [VALIDATION_STATUS, "frozen_at_training_end", frozen_roc, frozen_pr, 0.0],
            [VALIDATION_STATUS, "daily_past_only_refresh", rolling_roc, rolling_pr, rolling_pr - frozen_pr],
        ])
        single_feature_power(val_h, ml, spark)

        # Validation signal by requested operational groups.
        carrier_counts = {r[0]: int(r[1]) for r in lr_val.groupBy("Reporting_Airline").count().collect()}
        origin_counts = lr_val.groupBy("Origin").count().orderBy(F.desc("count")).collect()
        top20 = [r["Origin"] for r in origin_counts[:20] if int(r["count"]) >= 2000]
        excluded_rows = [["carrier", k, n, "fewer than 2000 flights"] for k, n in carrier_counts.items() if n < 2000]
        excluded_rows += [["origin", r["Origin"], int(r["count"]), "outside 20 largest or fewer than 2000 flights"] for r in origin_counts if r["Origin"] not in top20]
        signal_rows = group_signal(lr_val, "Reporting_Airline", "carrier")
        signal_rows += group_signal(lr_val, "Origin", "top20_origin", top20)
        signal_rows += group_signal(lr_val.withColumn("DepHour", F.col("ScheduledDepHour").cast("int")), "DepHour", "scheduled_departure_hour")
        signal_rows += group_signal(lr_val, "DayOfWeekCat", "weekday")
        write_csv(spark, "validation_signal_by_group.csv", ["Status", "GroupType", "Group", "Flights", "Delayed15Plus",
                  "Prevalence", "AUC_PR", "Top10PctPrecision", "Top10PctLift"], signal_rows)
        write_csv(spark, "validation_groups_excluded.csv", ["GroupType", "Group", "Flights", "Reason"], excluded_rows)

        coefficients(lr, train_vec, ml.feature_names(train_vec), spark)
        operating_rows = operating_thresholds(lr_val, lr_test, "LogisticRegression", spark)
        operating_rows += operating_thresholds(rf_val, rf_test, "RandomForest", spark)
        write_csv(spark, "operating_points.csv",
                  ["Status", "Model", "ValidationChosenOperatingPoint", "Threshold", "FlaggedShare", "Precision", "Recall", "AppliedSplit"], operating_rows)
        save_score_histograms({
            ("validation", "LogisticRegression"): lr_val,
            ("validation", "RandomForest"): rf_val,
            ("test", "LogisticRegression"): lr_test,
            ("test", "RandomForest"): rf_test,
        })
        print("DIAGNOSTICS_YARN_STATUS=SUCCEEDED", flush=True)
        return 0
    except Exception as exc:
        import traceback
        print(f"DIAGNOSTICS_YARN_STATUS=FAILED: {type(exc).__name__}: {exc}", flush=True)
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
