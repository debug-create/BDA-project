#!/usr/bin/env python3
"""Train and compare Spark MLlib classifiers for BTS flight arrival delays.

Target: ArrDel15 (1 if arrival delay >= 15 minutes, else 0).
Only features available before departure are used. FlightDate is used to
create a chronological train/test split, not included as a raw feature.
"""

import csv
import os
import sys
from datetime import datetime

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, functions as F

CLEAN_PATH = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS_PATH = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
HDFS_RESULTS = "hdfs://master:9000/user/root/flights/results/ml"
HDFS_MODELS = "hdfs://master:9000/user/root/flights/models"
LOCAL_RESULTS = "/results/ml"

TRAIN_DATE_FRACTION = 0.70
TOP_AIRPORTS = 50
RF_NUM_TREES = 30
RF_MAX_DEPTH = 6
RF_MAX_BINS = 64
LR_MAX_ITER = 20
SEED = 42

NUMERIC_FEATURES = [
    "Year", "Month", "DayofMonth", "DayOfWeek", "DayOfYear",
    "CRSDepTime", "CRSArrTime", "ScheduledDepartureHour",
    "ScheduledArrivalHour", "CRSElapsedTime", "Distance", "DistanceGroup",
]
CATEGORICAL_FEATURES = [
    "Reporting_Airline", "OriginBucket", "DestBucket", "DepTimeBlk", "HolidayPeriod",
]


def write_local_csv(filename, header, rows):
    os.makedirs(LOCAL_RESULTS, exist_ok=True)
    path = os.path.join(LOCAL_RESULTS, filename)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_hdfs_csv(spark, rows, filename, header):
    """Write a small result table to HDFS as a Spark CSV directory."""
    output_path = f"{HDFS_RESULTS}/{filename.replace('.csv', '')}"
    # All result tables are tiny (metrics or feature importance), so create the
    # DataFrame from the already-collected rows rather than re-scanning flights.
    data = [dict(zip(header, row)) for row in rows]
    result_df = spark.createDataFrame(data)
    (result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path))
    print(f"HDFS output: {output_path}", flush=True)


def feature_names_from_metadata(feature_df):
    """Recover VectorAssembler feature names; fall back to numbered names."""
    metadata = feature_df.schema["features"].metadata.get("ml_attr", {})
    attrs = metadata.get("attrs", {})
    num_attrs = int(metadata.get("num_attrs", 0))
    names = [f"feature_{i}" for i in range(num_attrs)]
    for attr_list in attrs.values():
        for item in attr_list:
            idx = int(item.get("idx", -1))
            if 0 <= idx < len(names):
                names[idx] = item.get("name", names[idx])
    return names


def metric_row(model_name, predictions):
    """Evaluate ranking and class-1 metrics on the same held-out test set."""
    auc_roc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc_pr_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0
    )
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    return [
        model_name,
        float(auc_roc_eval.evaluate(predictions)),
        float(auc_pr_eval.evaluate(predictions)),
        float(f1_eval.evaluate(predictions)),
        float(precision_eval.evaluate(predictions)),
        float(recall_eval.evaluate(predictions)),
        float(accuracy_eval.evaluate(predictions)),
    ]


def main() -> int:
    spark = SparkSession.builder.appName("BDA-03-Flight-Delay-MLlib").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    cached_frames = []

    try:
        print(f"Reading cleaned Parquet: {CLEAN_PATH}", flush=True)
        flights = spark.read.parquet(CLEAN_PATH)
        print(f"Reading holiday calendar: {HOLIDAYS_PATH}", flush=True)
        holidays = (
    spark.read.option("header", "true").csv(HOLIDAYS_PATH)
    .select("FlightDate", "HolidayPeriod")
    .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
    .filter(F.col("FlightDate").isNotNull())
)

        # Protect against duplicate calendar dates multiplying flight records.
        duplicate_dates = holidays.groupBy("FlightDate").count().filter(F.col("count") > 1).limit(1).count()
        if duplicate_dates:
            print("ERROR: holidays.csv contains duplicate FlightDate values.", flush=True)
            return 2

        # Remove cancelled/diverted flights and observations without the target.
        # Delay/outcome and delay-cause fields are intentionally not predictors.
        eligible = (
            flights.join(F.broadcast(holidays), on="FlightDate", how="left")
            .withColumn("HolidayPeriod", F.coalesce(F.col("HolidayPeriod"), F.lit("Regular day")))
            .filter(
                (F.col("Cancelled") == 0)
                & (F.col("Diverted") == 0)
                & F.col("ArrDel15").isin(0, 1)
            )
            .withColumn("label", F.col("ArrDel15").cast("double"))
            .withColumn("DayOfYear", F.dayofyear("FlightDate").cast("double"))
            .withColumn(
                "ScheduledDepartureHour",
                F.when(F.col("CRSDepTime").between(0, 2400),
                       F.pmod(F.floor(F.col("CRSDepTime") / 100), F.lit(24)).cast("double")),
            )
            .withColumn(
                "ScheduledArrivalHour",
                F.when(F.col("CRSArrTime").between(0, 2400),
                       F.pmod(F.floor(F.col("CRSArrTime") / 100), F.lit(24)).cast("double")),
            )
            .select(
                "FlightDate", "label", *NUMERIC_FEATURES,
                "Reporting_Airline", "Origin", "Dest", "DepTimeBlk", "HolidayPeriod",
            )
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached_frames.append(eligible)

        eligible_count = eligible.count()
        if eligible_count == 0:
            print("ERROR: no eligible flights remain after target/cancellation filters.", flush=True)
            return 3
        print(f"Eligible flights for ML (not cancelled/diverted, target non-null): {eligible_count:,}", flush=True)

        # Chronological split by complete flight dates (not random row split).
        # This avoids putting different flights from the same day into both sets.
        dates = [r[0] for r in eligible.select("FlightDate").distinct().orderBy("FlightDate").collect()]
        if len(dates) < 2:
            print("ERROR: fewer than two distinct flight dates; cannot make a date split.", flush=True)
            return 4
        split_index = max(1, min(int(len(dates) * TRAIN_DATE_FRACTION), len(dates) - 1))
        cutoff_date = dates[split_index]
        print(f"Distinct flight dates: {len(dates):,}", flush=True)
        print(f"Chronological train/test cutoff (test starts on this date): {cutoff_date}", flush=True)

        train_base = eligible.filter(F.col("FlightDate") < F.lit(cutoff_date)).persist(StorageLevel.MEMORY_AND_DISK)
        test_base = eligible.filter(F.col("FlightDate") >= F.lit(cutoff_date)).persist(StorageLevel.MEMORY_AND_DISK)
        cached_frames.extend([train_base, test_base])

        train_count = train_base.count()
        test_count = test_base.count()
        train_positive = int(train_base.agg(F.sum("label")).first()[0] or 0)
        test_positive = int(test_base.agg(F.sum("label")).first()[0] or 0)
        print(f"Training flights: {train_count:,}; delayed 15+ minutes: {train_positive:,} ({100.0 * train_positive / max(train_count, 1):.2f}%)", flush=True)
        print(f"Test flights: {test_count:,}; delayed 15+ minutes: {test_positive:,} ({100.0 * test_positive / max(test_count, 1):.2f}%)", flush=True)

        # Bound airport cardinality using the most frequent airports in training
        # data. Less frequent airports become OTHER; this keeps the model small
        # and allows trees to use a sensible number of feature bins.
        top_origins = [r[0] for r in train_base.groupBy("Origin").count().orderBy(F.desc("count"), F.asc("Origin")).limit(TOP_AIRPORTS).collect()]
        top_dests = [r[0] for r in train_base.groupBy("Dest").count().orderBy(F.desc("count"), F.asc("Dest")).limit(TOP_AIRPORTS).collect()]
        print(f"Airport grouping: top {TOP_AIRPORTS} origins and destinations retained; all others grouped as OTHER.", flush=True)

        features_base = (
            eligible
            .withColumn("OriginBucket", F.when(F.col("Origin").isin(top_origins), F.col("Origin")).otherwise(F.lit("OTHER")))
            .withColumn("DestBucket", F.when(F.col("Dest").isin(top_dests), F.col("Dest")).otherwise(F.lit("OTHER")))
            .withColumn("Reporting_Airline", F.coalesce(F.col("Reporting_Airline"), F.lit("UNKNOWN")))
            .withColumn("DepTimeBlk", F.coalesce(F.col("DepTimeBlk"), F.lit("UNKNOWN")))
            .withColumn("HolidayPeriod", F.coalesce(F.col("HolidayPeriod"), F.lit("Regular day")))
            .select("FlightDate", "label", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
        )
        train = features_base.filter(F.col("FlightDate") < F.lit(cutoff_date))
        test = features_base.filter(F.col("FlightDate") >= F.lit(cutoff_date))

        imputed_numeric = [f"{name}__imputed" for name in NUMERIC_FEATURES]
        index_cols = [f"{name}__idx" for name in CATEGORICAL_FEATURES]
        encoded_cols = [f"{name}__ohe" for name in CATEGORICAL_FEATURES]
        stages = [Imputer(strategy="median", inputCols=NUMERIC_FEATURES, outputCols=imputed_numeric)]
        stages.extend(
            StringIndexer(inputCol=name, outputCol=index_name, handleInvalid="keep")
            for name, index_name in zip(CATEGORICAL_FEATURES, index_cols)
        )
        stages.append(OneHotEncoder(inputCols=index_cols, outputCols=encoded_cols, handleInvalid="keep", dropLast=False))
        stages.append(VectorAssembler(inputCols=imputed_numeric + encoded_cols, outputCol="features", handleInvalid="keep"))
        feature_pipeline = Pipeline(stages=stages)

        print("Fitting feature preprocessing on training data only...", flush=True)
        feature_pipeline_model = feature_pipeline.fit(train)
        train_vec = feature_pipeline_model.transform(train).select("label", "features").persist(StorageLevel.MEMORY_AND_DISK)
        test_vec = feature_pipeline_model.transform(test).select("label", "features").persist(StorageLevel.MEMORY_AND_DISK)
        cached_frames.extend([train_vec, test_vec])
        # Materialize once so both model fits can reuse the transformed rows.
        train_vec_count = train_vec.count()
        test_vec_count = test_vec.count()
        print(f"Feature vectors materialized: train={train_vec_count:,}, test={test_vec_count:,}", flush=True)

        metrics = []
        confusion_rows = []

        print("Training Logistic Regression baseline...", flush=True)
        lr = LogisticRegression(
            labelCol="label", featuresCol="features", predictionCol="prediction",
            rawPredictionCol="rawPrediction", probabilityCol="probability",
            maxIter=LR_MAX_ITER, regParam=0.01, elasticNetParam=0.0, standardization=True,
        )
        lr_model = lr.fit(train_vec)
        lr_predictions = lr_model.transform(test_vec).select("label", "prediction", "rawPrediction", "probability").persist(StorageLevel.MEMORY_AND_DISK)
        cached_frames.append(lr_predictions)
        lr_metrics = metric_row("LogisticRegression", lr_predictions)
        metrics.append(lr_metrics)
        print("Logistic Regression metrics: AUC_ROC={:.5f}, AUC_PR={:.5f}, F1_class1={:.5f}, precision_class1={:.5f}, recall_class1={:.5f}, accuracy={:.5f}".format(*lr_metrics[1:]), flush=True)
        lr_confusion = lr_predictions.groupBy("label", "prediction").count().collect()
        for row in lr_confusion:
            confusion_rows.append(["LogisticRegression", int(row["label"]), int(row["prediction"]), int(row["count"])])
        lr_predictions.unpersist()
        cached_frames.remove(lr_predictions)

        print("Training Random Forest classifier...", flush=True)
        rf = RandomForestClassifier(
            labelCol="label", featuresCol="features", predictionCol="prediction",
            rawPredictionCol="rawPrediction", probabilityCol="probability",
            numTrees=RF_NUM_TREES, maxDepth=RF_MAX_DEPTH, maxBins=RF_MAX_BINS,
            featureSubsetStrategy="sqrt", seed=SEED,
        )
        rf_model = rf.fit(train_vec)
        rf_predictions = rf_model.transform(test_vec).select("label", "prediction", "rawPrediction", "probability").persist(StorageLevel.MEMORY_AND_DISK)
        cached_frames.append(rf_predictions)
        rf_metrics = metric_row("RandomForest", rf_predictions)
        metrics.append(rf_metrics)
        print("Random Forest metrics: AUC_ROC={:.5f}, AUC_PR={:.5f}, F1_class1={:.5f}, precision_class1={:.5f}, recall_class1={:.5f}, accuracy={:.5f}".format(*rf_metrics[1:]), flush=True)
        rf_confusion = rf_predictions.groupBy("label", "prediction").count().collect()
        for row in rf_confusion:
            confusion_rows.append(["RandomForest", int(row["label"]), int(row["prediction"]), int(row["count"])])

        # Translate feature importances back to assembled feature names.
        names = feature_names_from_metadata(train_vec)
        importances = rf_model.featureImportances.toArray().tolist()
        importance_rows = sorted(
            [[names[i] if i < len(names) else f"feature_{i}", float(value)] for i, value in enumerate(importances)],
            key=lambda row: row[1], reverse=True,
        )[:30]

        # Persist metrics, predictions' confusion counts, class counts and top
        # RF features on the host results mount and under the HDFS results path.
        metrics_header = ["Model", "AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1", "Recall_class1", "Accuracy"]
        local_metrics = write_local_csv("metrics.csv", metrics_header, metrics)
        write_hdfs_csv(spark, metrics, "metrics.csv", metrics_header)
        print(f"Local metrics: {local_metrics}", flush=True)

        confusion_header = ["Model", "ActualLabel", "PredictedLabel", "Count"]
        local_confusion = write_local_csv("confusion_counts.csv", confusion_header, confusion_rows)
        write_hdfs_csv(spark, confusion_rows, "confusion_counts.csv", confusion_header)
        print(f"Local confusion counts: {local_confusion}", flush=True)

        importance_header = ["Feature", "Importance"]
        local_importance = write_local_csv("random_forest_feature_importance_top30.csv", importance_header, importance_rows)
        write_hdfs_csv(spark, importance_rows, "random_forest_feature_importance_top30.csv", importance_header)
        print(f"Local top feature importances: {local_importance}", flush=True)

        class_rows = [
            ["train", train_count, train_positive, train_count - train_positive, cutoff_date.isoformat()],
            ["test", test_count, test_positive, test_count - test_positive, cutoff_date.isoformat()],
        ]
        class_header = ["Split", "Flights", "Delayed15Plus", "NotDelayed15Plus", "TestStartsOn"]
        local_classes = write_local_csv("class_counts.csv", class_header, class_rows)
        write_hdfs_csv(spark, class_rows, "class_counts.csv", class_header)
        print(f"Local class counts: {local_classes}", flush=True)

        print("Saving fitted preprocessing pipeline and classifiers to HDFS...", flush=True)
        feature_pipeline_model.write().overwrite().save(f"{HDFS_MODELS}/feature_pipeline")
        lr_model.write().overwrite().save(f"{HDFS_MODELS}/logistic_regression")
        rf_model.write().overwrite().save(f"{HDFS_MODELS}/random_forest")
        print(f"Saved feature pipeline and models under {HDFS_MODELS}", flush=True)

        summary_path = os.path.join(LOCAL_RESULTS, "run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as summary:
            summary.write("BDA MLlib flight-arrival-delay classification\n")
            summary.write(f"Target: ArrDel15; positive class: arrival delay >= 15 minutes\n")
            summary.write("Cancelled and diverted flights excluded; non-null target required.\n")
            summary.write("Predictors are pre-departure schedule, carrier, route buckets and calendar/holiday fields only.\n")
            summary.write(f"Eligible records: {eligible_count}\n")
            summary.write(f"Distinct dates: {len(dates)}\n")
            summary.write(f"Chronological test start date: {cutoff_date.isoformat()}\n")
            summary.write(f"Training records: {train_count}; positive: {train_positive}\n")
            summary.write(f"Test records: {test_count}; positive: {test_positive}\n")
            summary.write(f"Random Forest settings: numTrees={RF_NUM_TREES}, maxDepth={RF_MAX_DEPTH}, maxBins={RF_MAX_BINS}\n")
            summary.write("Metrics are measured on the same held-out chronological test set; F1, precision, and recall refer to class 1.\n\n")
            for row in metrics:
                summary.write(", ".join([row[0]] + [f"{value:.6f}" for value in row[1:]]) + "\n")
        print(f"Local run summary: {summary_path}", flush=True)
        print(f"ML_ELIGIBLE_ROWS={eligible_count}", flush=True)
        print(f"ML_TRAIN_ROWS={train_count}", flush=True)
        print(f"ML_TEST_ROWS={test_count}", flush=True)
        print(f"ML_TEST_START_DATE={cutoff_date.isoformat()}", flush=True)
        print("MLLIB_JOB_STATUS=SUCCEEDED", flush=True)
        print(f"MLLIB_FINISHED_UTC={datetime.utcnow().isoformat(timespec='seconds')}Z", flush=True)
        rf_predictions.unpersist()
        cached_frames.remove(rf_predictions)
        return 0
    except Exception as error:
        print(f"ERROR: MLlib pipeline failed: {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        for frame in reversed(cached_frames):
            try:
                frame.unpersist()
            except Exception:
                pass
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
