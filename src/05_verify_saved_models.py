#!/usr/bin/env python3
"""Load the persisted Spark ML artifacts and fail if any artifact is unreadable."""

import argparse

from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
)
from pyspark.sql import SparkSession


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-uri",
        required=True,
        help="Model root written by 03_ml.py (for example hdfs:///.../models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.models_uri.rstrip("/")
    spark = SparkSession.builder.appName("BDA-05-Verify-Saved-Models").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    pipeline = PipelineModel.load(f"{root}/feature_pipeline")
    logistic = LogisticRegressionModel.load(f"{root}/logistic_regression")
    forest = RandomForestClassificationModel.load(f"{root}/random_forest")

    if not pipeline.stages:
        raise RuntimeError("The saved preprocessing PipelineModel contains no stages.")
    if logistic.numFeatures <= 0 or forest.numFeatures <= 0:
        raise RuntimeError("A saved classifier reports no input features.")
    if logistic.numFeatures != forest.numFeatures:
        raise RuntimeError(
            "Saved classifiers disagree on feature-vector width: "
            f"LR={logistic.numFeatures}, RF={forest.numFeatures}"
        )

    print(f"MODEL_ROOT={root}")
    print(f"PREPROCESSING_STAGES={len(pipeline.stages)}")
    print(f"MODEL_FEATURES={logistic.numFeatures}")
    print(f"RANDOM_FOREST_TREES={forest.getNumTrees}")
    print("SAVED_MODEL_VERIFICATION_STATUS=PASSED")
    spark.stop()


if __name__ == "__main__":
    main()
