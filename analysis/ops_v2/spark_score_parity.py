#!/usr/bin/env python3
"""Compute Spark evaluator metrics for host-written validation score Parquet."""

import argparse
import json

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession, functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    options = parser.parse_args()
    spark = (SparkSession.builder.appName("OpsV2ScoreParity")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    rows = []
    try:
        for path in options.paths:
            frame = spark.read.parquet(path).select(
                F.col("label").cast("double").alias("label"),
                F.col("score").cast("double").alias("score"))
            pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score",
                                                metricName="areaUnderPR").evaluate(frame)
            roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score",
                                                 metricName="areaUnderROC").evaluate(frame)
            rows.append({"path": path, "rows": frame.count(), "spark_pr_auc": pr, "spark_roc_auc": roc})
        print("SPARK_SCORE_PARITY=" + json.dumps(rows, sort_keys=True), flush=True)
        print("SPARK_SCORE_PARITY_STATUS=SUCCEEDED", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
