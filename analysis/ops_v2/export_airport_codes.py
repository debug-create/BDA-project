#!/usr/bin/env python3
"""Export the distinct IATA airport codes present in the existing clean Parquet."""

from pyspark.sql import SparkSession, functions as F

CLEAN = "hdfs://master:9000/user/root/flights/clean"
OUTPUT = "file:///results/experiments/ml_redesign/ops_v2/airport_codes"


def main():
    spark = (SparkSession.builder.appName("BDA-OpsV2-Airport-Codes")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        flights = spark.read.parquet(CLEAN)
        codes = (flights.select(F.col("Origin").alias("Airport"))
                 .union(flights.select(F.col("Dest").alias("Airport")))
                 .filter(F.col("Airport").isNotNull())
                 .distinct().orderBy("Airport"))
        count = codes.count()
        codes.coalesce(1).write.mode("overwrite").option("header", True).csv(OUTPUT)
        print(f"AIRPORT_CODES_STATUS=SUCCEEDED count={count} output={OUTPUT}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
