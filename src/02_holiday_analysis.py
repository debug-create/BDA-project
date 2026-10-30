#!/usr/bin/env python3
"""Holiday-vs-regular flight delay analysis using Spark SQL on YARN.

Chosen date windows (editable in holidays.csv):
- Halloween: 2024-10-31
- Thanksgiving week: 2024-11-25 through 2024-12-01
- Christmas/New Year period: 2024-12-20 through 2024-12-31

Cancellation rate uses all scheduled flight records. Average departure delay
uses non-cancelled, non-diverted flights with a valid DepDelay. Arrival-delay-
15+ percentage uses non-cancelled/non-diverted flights with non-null ArrDel15.
Delay-cause fields are descriptive only and are not ML features.
"""

import csv
import os
import sys
from datetime import datetime

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F, types as T

CLEAN_PATH = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS_PATH = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
HDFS_RESULTS = "hdfs://master:9000/user/root/flights/results/holiday_analysis"
LOCAL_RESULTS = "/results/holiday_analysis"
MIN_GROUP_FLIGHTS = 100

OUTPUTS = {
    "overall_comparison": "overall_comparison",
    "daily_trends": "daily_trends",
    "by_airline": "by_airline",
    "by_origin_airport": "by_origin_airport",
    "delay_cause_share": "delay_cause_share",
    "weekday_hour_heatmap": "weekday_hour_heatmap",
}

HOLIDAY_SCHEMA = T.StructType([
    T.StructField("FlightDate", T.StringType(), True),
    T.StructField("HolidayPeriod", T.StringType(), True),
])


def write_outputs(spark, df, name):
    """Write an aggregated table to HDFS CSV and a single CSV on the host mount."""
    # These are small, already-aggregated result tables (never raw flights).
    rows = df.collect()
    os.makedirs(LOCAL_RESULTS, exist_ok=True)
    local_path = os.path.join(LOCAL_RESULTS, f"{name}.csv")
    with open(local_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        for row in rows:
            writer.writerow([row[column] for column in df.columns])

    # Re-create the small table from collected rows so the large source DataFrame
    # does not need to be scanned again just to save the same result to HDFS.
    hdfs_path = f"{HDFS_RESULTS}/{name}"
    result_df = spark.createDataFrame(rows, schema=df.schema)
    result_df.write.mode("overwrite").option("header", "true").csv(hdfs_path)
    print(f"Wrote {len(rows):,} aggregated rows: {local_path}", flush=True)
    print(f"HDFS output: {hdfs_path}", flush=True)


def main() -> int:
    spark = SparkSession.builder.appName("BDA-02-Holiday-Analysis").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Reading cleaned Parquet: {CLEAN_PATH}", flush=True)
        flights = spark.read.parquet(CLEAN_PATH)
        print(f"Reading holiday calendar from HDFS: {HOLIDAYS_PATH}", flush=True)
        holidays = (
            spark.read.option("header", "true")
            .schema(HOLIDAY_SCHEMA)
            .csv(HOLIDAYS_PATH)
            .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
            .filter(F.col("FlightDate").isNotNull())
        )

        # Prevent duplicated calendar dates from duplicating flight records.
        duplicate_dates = holidays.groupBy("FlightDate").count().filter(F.col("count") > 1).limit(1).count()
        if duplicate_dates:
            print("ERROR: holidays.csv contains duplicate FlightDate values.", flush=True)
            return 2

        enriched = (
            flights.join(F.broadcast(holidays), on="FlightDate", how="left")
            .withColumn("HolidayPeriod", F.coalesce(F.col("HolidayPeriod"), F.lit("Regular day")))
            .withColumn(
                "ScheduledDepartureHour",
                F.when(
                    F.col("CRSDepTime").between(0, 2359),
                    F.floor(F.col("CRSDepTime") / F.lit(100)).cast("int"),
                ),
            )
        ).persist(StorageLevel.MEMORY_AND_DISK)
        # Materialize once and reuse across SQL aggregates to avoid rescanning Parquet.
        total_flights = enriched.count()
        enriched.createOrReplaceTempView("flights_with_holidays")

        # Metric definitions are deliberately explicit for interpretation in viva.
        overall = spark.sql("""
            SELECT
                HolidayPeriod,
                CASE WHEN HolidayPeriod = 'Regular day' THEN 'Regular day' ELSE 'Holiday period' END AS DayType,
                COUNT(*) AS Flights,
                SUM(CASE WHEN Cancelled = 1 THEN 1 ELSE 0 END) AS CancelledFlights,
                ROUND(100.0 * SUM(CASE WHEN Cancelled = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS CancellationRatePct,
                ROUND(AVG(CASE WHEN Cancelled = 0 AND Diverted = 0 THEN DepDelay END), 2) AS AvgDepartureDelayMinutes,
                SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END) AS EligibleArrivalFlights,
                SUM(CASE WHEN ArrDel15 = 1 AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END) AS ArrivalDelayed15PlusFlights,
                CASE WHEN SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END) > 0
                    THEN ROUND(100.0 * SUM(CASE WHEN ArrDel15 = 1 AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END)
                    / SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END), 3)
                    ELSE NULL END AS ArrivalDelay15PlusPct
            FROM flights_with_holidays
            GROUP BY HolidayPeriod
            ORDER BY CASE WHEN HolidayPeriod = 'Regular day' THEN 0 ELSE 1 END, HolidayPeriod
        """)

        daily = spark.sql("""
            SELECT
                FlightDate,
                HolidayPeriod,
                COUNT(*) AS Flights,
                ROUND(AVG(CASE WHEN Cancelled = 0 AND Diverted = 0 THEN DepDelay END), 2) AS AvgDepartureDelayMinutes,
                ROUND(100.0 * SUM(CASE WHEN Cancelled = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS CancellationRatePct,
                CASE WHEN SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END) > 0
                    THEN ROUND(100.0 * SUM(CASE WHEN ArrDel15 = 1 AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END)
                    / SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END), 3)
                    ELSE NULL END AS ArrivalDelay15PlusPct
            FROM flights_with_holidays
            GROUP BY FlightDate, HolidayPeriod
            ORDER BY FlightDate
        """)

        by_airline = spark.sql(f"""
            SELECT
                Reporting_Airline,
                HolidayPeriod,
                COUNT(*) AS Flights,
                ROUND(AVG(CASE WHEN Cancelled = 0 AND Diverted = 0 THEN DepDelay END), 2) AS AvgDepartureDelayMinutes,
                ROUND(100.0 * SUM(CASE WHEN Cancelled = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS CancellationRatePct,
                CASE WHEN SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END) > 0
                    THEN ROUND(100.0 * SUM(CASE WHEN ArrDel15 = 1 AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END)
                    / SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END), 3)
                    ELSE NULL END AS ArrivalDelay15PlusPct
            FROM flights_with_holidays
            GROUP BY Reporting_Airline, HolidayPeriod
            HAVING COUNT(*) >= {MIN_GROUP_FLIGHTS}
            ORDER BY Reporting_Airline, HolidayPeriod
        """)

        by_origin = spark.sql(f"""
            SELECT
                Origin AS OriginAirport,
                HolidayPeriod,
                COUNT(*) AS Flights,
                ROUND(AVG(CASE WHEN Cancelled = 0 AND Diverted = 0 THEN DepDelay END), 2) AS AvgDepartureDelayMinutes,
                ROUND(100.0 * SUM(CASE WHEN Cancelled = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS CancellationRatePct,
                CASE WHEN SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END) > 0
                    THEN ROUND(100.0 * SUM(CASE WHEN ArrDel15 = 1 AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END)
                    / SUM(CASE WHEN ArrDel15 IS NOT NULL AND Cancelled = 0 AND Diverted = 0 THEN 1 ELSE 0 END), 3)
                    ELSE NULL END AS ArrivalDelay15PlusPct
            FROM flights_with_holidays
            GROUP BY Origin, HolidayPeriod
            HAVING COUNT(*) >= {MIN_GROUP_FLIGHTS}
            ORDER BY Origin, HolidayPeriod
        """)

        cause_union = spark.sql("""
            SELECT 'CarrierDelay' AS DelayCause, SUM(COALESCE(CarrierDelay, 0.0)) AS DelayMinutes FROM flights_with_holidays
            UNION ALL
            SELECT 'WeatherDelay', SUM(COALESCE(WeatherDelay, 0.0)) FROM flights_with_holidays
            UNION ALL
            SELECT 'NASDelay', SUM(COALESCE(NASDelay, 0.0)) FROM flights_with_holidays
            UNION ALL
            SELECT 'SecurityDelay', SUM(COALESCE(SecurityDelay, 0.0)) FROM flights_with_holidays
            UNION ALL
            SELECT 'LateAircraftDelay', SUM(COALESCE(LateAircraftDelay, 0.0)) FROM flights_with_holidays
        """)
        cause_union.createOrReplaceTempView("delay_cause_totals")
        cause_shares = spark.sql("""
            SELECT DelayCause, DelayMinutes,
                CASE WHEN SUM(DelayMinutes) OVER () > 0
                    THEN ROUND(100.0 * DelayMinutes / SUM(DelayMinutes) OVER (), 3)
                    ELSE NULL END AS ShareOfRecordedDelayCauseMinutesPct
            FROM delay_cause_totals
            ORDER BY DelayMinutes DESC
        """)

        weekday_hour = spark.sql("""
            SELECT
                DayOfWeek,
                ScheduledDepartureHour,
                COUNT(*) AS Flights,
                ROUND(AVG(CASE WHEN Cancelled = 0 AND Diverted = 0 THEN DepDelay END), 2) AS AvgDepartureDelayMinutes
            FROM flights_with_holidays
            WHERE ScheduledDepartureHour BETWEEN 0 AND 23
            GROUP BY DayOfWeek, ScheduledDepartureHour
            ORDER BY DayOfWeek, ScheduledDepartureHour
        """)

        outputs = [
            (overall, "overall_comparison"),
            (daily, "daily_trends"),
            (by_airline, "by_airline"),
            (by_origin, "by_origin_airport"),
            (cause_shares, "delay_cause_share"),
            (weekday_hour, "weekday_hour_heatmap"),
        ]
        print(f"Minimum airline/airport group size: {MIN_GROUP_FLIGHTS} flights", flush=True)
        for data, name in outputs:
            write_outputs(spark, data, name)

        # Report broad record counts for a reproducible run log.
        holiday_flights = enriched.filter(F.col("HolidayPeriod") != "Regular day").count()
        print(f"Total cleaned flights analysed: {total_flights:,}", flush=True)
        print(f"Flights falling in selected holiday windows: {holiday_flights:,}", flush=True)
        print(f"Holiday calendar dates: {holidays.count():,}", flush=True)
        print(f"HOLIDAY_ANALYSIS_STATUS=SUCCEEDED", flush=True)
        print(f"HOLIDAY_ANALYSIS_FINISHED_UTC={datetime.utcnow().isoformat(timespec='seconds')}Z", flush=True)
        enriched.unpersist()
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
