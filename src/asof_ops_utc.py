"""UTC-correct conservative hourly operations features.

Airport/date/hour UTC offsets come from the checked-in build recipe in
``analysis/ops_v2/build_airport_tz.py``.  Ambiguous DST hours use the later UTC
interpretation for events and the earlier UTC interpretation for targets.  That
can discard a valid event but cannot admit an event that occurs after T.
"""

from pyspark.sql import functions as F, types as T

from asof_ops import JOIN_KEYS, OPS_FEATURES, SMOOTHING


TZ_HDFS_PATH = "hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/reference/airport_tz.csv"
REQUIRED_UTC_EVENT_COLUMNS = [
    "DepDelay", "ArrDelay", "Cancelled", "Diverted", "CRSDepTime",
    "CRSElapsedTime", "FlightDate", "Reporting_Airline", "Origin", "Dest",
]
ORIGIN_ONLY_FEATURES = OPS_FEATURES[:5]
ORIGIN_INBOUND_FEATURES = OPS_FEATURES[:8]


def minute_of_day(column):
    value = column.cast("int")
    hour = F.floor(value / 100).cast("int")
    minute = (value % 100).cast("int")
    return F.when(value == 2400, F.lit(0)).when(
        value.isNotNull() & hour.between(0, 23) & minute.between(0, 59), hour * 60 + minute)


def timezone_schema():
    fields = [
        T.StructField("Airport", T.StringType(), False),
        T.StructField("LocalDate", T.DateType(), False),
        T.StructField("TimeZone", T.StringType(), False),
        T.StructField("MatchStatus", T.StringType(), False),
    ]
    fields.extend(T.StructField(f"OffsetFold{fold}H{hour:02d}", T.IntegerType(), True)
                  for fold in (0, 1) for hour in range(24))
    return T.StructType(fields)


def load_timezone_offsets(spark, path=TZ_HDFS_PATH):
    table = spark.read.option("header", True).schema(timezone_schema()).csv(path)
    duplicate = table.groupBy("Airport", "LocalDate").count().filter(F.col("count") != 1).limit(1).count()
    if duplicate:
        raise ValueError("airport timezone table is not unique by Airport, LocalDate")
    return table


def _offset_array(prefix):
    return F.array(*[F.col(f"{prefix}H{hour:02d}") for hour in range(24)])


def _offset_for_hour(prefix, hour_column):
    return F.element_at(_offset_array(prefix), hour_column.cast("int") + F.lit(1))


def _empty_int_array():
    return F.array().cast("array<int>")


def _sequence_if(start, end, condition):
    return F.when(condition, F.sequence(start.cast("int"), end.cast("int"))).otherwise(_empty_int_array())


def add_utc_event_times(flights, timezone_table):
    """Convert local scheduled/actual departure and arrival times to UTC hours."""
    dep_min = minute_of_day(F.col("CRSDepTime"))
    day = F.datediff("FlightDate", F.lit("1970-01-01").cast("date"))
    base = (flights.withColumn("_dep_min", dep_min)
            .withColumn("_dep_hod", F.floor(dep_min / 60).cast("int")))
    tz = timezone_table.alias("tz")
    joined = base.alias("f").join(
        F.broadcast(tz),
        (F.col("f.Origin") == F.col("tz.Airport")) & (F.col("f.FlightDate") == F.col("tz.LocalDate")),
        "left")
    fold0 = _offset_for_hour("OffsetFold0", F.col("_dep_hod"))
    fold1 = _offset_for_hour("OffsetFold1", F.col("_dep_hod"))
    # UTC = local - offset.  The smaller offset gives the later conservative
    # event instant; the larger gives the earlier conservative target instant.
    event_offset = F.least(fold0, fold1)
    target_offset = F.greatest(fold0, fold1)
    local_sched_minute = day * F.lit(1440) + F.col("_dep_min")
    sched_event_utc = local_sched_minute - event_offset
    sched_target_utc = local_sched_minute - target_offset
    return (joined
            .withColumn("_tz_matched", event_offset.isNotNull().cast("int"))
            .withColumn("_utc_offset_event_min", event_offset)
            .withColumn("_utc_offset_target_min", target_offset)
            .withColumn("_sched_dep_utc_event_min", sched_event_utc)
            .withColumn("_sched_dep_utc_target_min", sched_target_utc)
            .withColumn("_sched_dep_utc_hour", F.floor(sched_event_utc / 60).cast("long"))
            .withColumn("_actual_dep_utc_hour",
                        F.floor((sched_event_utc + F.col("DepDelay")) / 60).cast("long"))
            .withColumn("_sched_arr_utc_hour",
                        F.floor((sched_event_utc + F.col("CRSElapsedTime")) / 60).cast("long"))
            .withColumn("_actual_arr_utc_hour",
                        F.floor((sched_event_utc + F.col("CRSElapsedTime") + F.col("ArrDelay")) / 60)
                        .cast("long"))
            .select("f.*", "_tz_matched", "_utc_offset_event_min", "_utc_offset_target_min",
                    "_sched_dep_utc_event_min", "_sched_dep_utc_target_min", "_sched_dep_utc_hour",
                    "_actual_dep_utc_hour", "_sched_arr_utc_hour", "_actual_arr_utc_hour"))


def _outcome_priors(events, cutoff):
    train = events.filter(F.col("FlightDate") < F.lit(cutoff))
    departures = train.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) & F.col("DepDelay").isNotNull())
    arrivals = train.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) & F.col("ArrDelay").isNotNull())
    dep = departures.agg(F.avg("DepDelay").alias("mean"),
                         F.avg((F.col("DepDelay") >= 15).cast("double")).alias("share")).first()
    arr = arrivals.agg(F.avg("ArrDelay").alias("mean"),
                       F.avg((F.col("ArrDelay") >= 15).cast("double")).alias("share")).first()
    return {"dep_mean": float(dep["mean"]), "dep_share": float(dep["share"]),
            "arr_mean": float(arr["mean"]), "arr_share": float(arr["share"])}


def _departed_window(events):
    valid = events.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
                          F.col("DepDelay").isNotNull() & F.col("_actual_dep_utc_hour").isNotNull())
    hourly = (valid.groupBy(F.col("Origin").alias("_airport"),
                            F.col("_actual_dep_utc_hour").alias("_event_hour"))
              .agg(F.count("*").alias("_dep_n"), F.sum("DepDelay").alias("_dep_sum"),
                   F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_dep15")))
    return (hourly.withColumn("_t_utc", F.explode(F.sequence(F.col("_event_hour") + 1,
                                                              F.col("_event_hour") + 3)))
            .groupBy("_airport", "_t_utc")
            .agg(F.sum("_dep_n").alias("_dep_n"), F.sum("_dep_sum").alias("_dep_sum"),
                 F.sum("_dep15").alias("_dep15")))


def _landed_window(events):
    valid = events.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
                          F.col("ArrDelay").isNotNull() & F.col("_actual_arr_utc_hour").isNotNull())
    hourly = (valid.groupBy(F.col("Dest").alias("_airport"),
                            F.col("_actual_arr_utc_hour").alias("_event_hour"))
              .agg(F.count("*").alias("_arr_n"), F.sum("ArrDelay").alias("_arr_sum"),
                   F.sum((F.col("ArrDelay") >= 15).cast("long")).alias("_arr15")))
    return (hourly.withColumn("_t_utc", F.explode(F.sequence(F.col("_event_hour") + 1,
                                                              F.col("_event_hour") + 3)))
            .groupBy("_airport", "_t_utc")
            .agg(F.sum("_arr_n").alias("_arr_n"), F.sum("_arr_sum").alias("_arr_sum"),
                 F.sum("_arr15").alias("_arr15")))


def _origin_backlog(events):
    expanded = events.filter(F.col("_sched_dep_utc_hour").isNotNull()).select(
        F.col("Origin").alias("_airport"), "Cancelled", "Diverted", "_actual_dep_utc_hour",
        F.explode(F.sequence(F.col("_sched_dep_utc_hour") + 1,
                             F.col("_sched_dep_utc_hour") + 3)).alias("_t_utc"))
    backlog = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
               F.col("_actual_dep_utc_hour").isNull() | (F.col("_actual_dep_utc_hour") >= F.col("_t_utc")))
    return (expanded.groupBy("_airport", "_t_utc")
            .agg(F.count("*").alias("_sched_n"), F.sum(backlog.cast("long")).alias("_backlog_n")))


def _carrier_departed(events):
    valid = events.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
                          F.col("DepDelay").isNotNull() & F.col("_actual_dep_utc_hour").isNotNull())
    event_day = F.floor(F.col("_actual_dep_utc_hour") / 24).cast("long")
    event_hour = F.pmod(F.col("_actual_dep_utc_hour"), F.lit(24)).cast("int")
    hourly = (valid.groupBy("Reporting_Airline", event_day.alias("_t_day"), event_hour.alias("_event_hod"))
              .agg(F.count("*").alias("_dep_n"), F.sum("DepDelay").alias("_dep_sum"),
                   F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_dep15")))
    expanded = hourly.withColumn(
        "_t_hod", F.explode(_sequence_if(F.col("_event_hod") + 1, F.lit(23), F.col("_event_hod") < 23)))
    return (expanded.groupBy("Reporting_Airline", "_t_day", "_t_hod")
            .agg(F.sum("_dep_n").alias("_carrier_dep_n"), F.sum("_dep_sum").alias("_carrier_dep_sum"),
                 F.sum("_dep15").alias("_carrier_dep15")))


def _carrier_backlog(events):
    valid = events.filter(F.col("_sched_dep_utc_hour").isNotNull())
    sched_day = F.floor(F.col("_sched_dep_utc_hour") / 24).cast("long")
    sched_hour = F.pmod(F.col("_sched_dep_utc_hour"), F.lit(24)).cast("int")
    base = valid.select("Reporting_Airline", "Cancelled", "Diverted", "_actual_dep_utc_hour",
                        sched_day.alias("_t_day"), sched_hour.alias("_sched_hod"))
    expanded = base.withColumn(
        "_t_hod", F.explode(_sequence_if(F.col("_sched_hod") + 1, F.lit(23), F.col("_sched_hod") < 23)))
    snapshot = F.col("_t_day") * 24 + F.col("_t_hod")
    backlog = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
               F.col("_actual_dep_utc_hour").isNull() | (F.col("_actual_dep_utc_hour") >= snapshot))
    return (expanded.groupBy("Reporting_Airline", "_t_day", "_t_hod")
            .agg(F.count("*").alias("_carrier_sched_n"),
                 F.sum(backlog.cast("long")).alias("_carrier_backlog_n")))


def _materialized_join(targets, block, condition, output_columns, label, keep, release):
    expected = targets.count()
    joined = targets.join(block, condition, "left")
    actual = joined.count()
    if actual != expected:
        raise ValueError(f"{label}: row count changed from {expected:,} to {actual:,}")
    print(f"ASOF_UTC_ROW_COUNT_CHECK=PASSED {label} rows={actual}", flush=True)
    next_targets = keep(joined.select(*[targets[column] for column in targets.columns], *output_columns))
    next_targets.count(); release(targets); release(block)
    return next_targets


def attach_asof_features_utc(event_source, eligible, lead_minutes, prior_cutoff, timezone_table,
                             keep, release, priors=None):
    """Attach all 16 UTC-ordered operational features and return coverage metadata."""
    if lead_minutes not in (60, 120, 180):
        raise ValueError("lead_minutes must be 60, 120, or 180")
    missing = [column for column in REQUIRED_UTC_EVENT_COLUMNS if column not in event_source.columns]
    if missing:
        raise ValueError(f"UTC asof_ops requires clean columns: {missing}")
    expected_events = event_source.count()
    events = keep(add_utc_event_times(event_source, timezone_table).select(
        "FlightDate", "Reporting_Airline", "Origin", "Dest", "Cancelled", "Diverted",
        "DepDelay", "ArrDelay", "_tz_matched", "_sched_dep_utc_hour", "_actual_dep_utc_hour",
        "_sched_arr_utc_hour", "_actual_arr_utc_hour"))
    actual_events = events.count()
    if actual_events != expected_events:
        raise ValueError(f"timezone join changed event rows {expected_events:,} -> {actual_events:,}")
    coverage_row = events.agg(F.sum("_tz_matched").alias("matched")).first()
    matched_events = int(coverage_row["matched"] or 0)
    fitted_priors = dict(priors) if priors is not None else _outcome_priors(events, prior_cutoff)

    # Eligible rows only retain scheduled hour.  With integer-hour US offsets,
    # the minute within the hour does not affect the conservative target hour.
    target_base = (eligible.select(*JOIN_KEYS).distinct()
                   .withColumn("_flight_day", F.datediff("FlightDate", F.lit("1970-01-01").cast("date")))
                   .withColumn("_dep_hod", F.col("ScheduledDepHour").cast("int")))
    tz = timezone_table.alias("tz")
    target_join = target_base.alias("f").join(
        F.broadcast(tz),
        (F.col("f.Origin") == F.col("tz.Airport")) & (F.col("f.FlightDate") == F.col("tz.LocalDate")),
        "left")
    target_offset = F.greatest(_offset_for_hour("OffsetFold0", F.col("_dep_hod")),
                               _offset_for_hour("OffsetFold1", F.col("_dep_hod")))
    t_minute = (F.col("_flight_day") * 1440 + F.col("_dep_hod") * 60 - target_offset - lead_minutes)
    targets = keep(target_join.select(
        *[F.col(f"f.{column}").alias(column) for column in JOIN_KEYS],
        F.floor(t_minute / 60).cast("long").alias("_t_utc"),
        target_offset.isNotNull().cast("int").alias("_target_tz_matched"),
        F.floor(F.floor(t_minute / 60) / 24).cast("long").alias("_t_day"),
        F.pmod(F.floor(t_minute / 60), F.lit(24)).cast("int").alias("_t_hod")))
    target_rows = targets.count()
    target_matched = int(targets.agg(F.sum("_target_tz_matched")).first()[0] or 0)

    departure_events = events.filter(F.col("DepDelay").isNull() | (F.col("DepDelay") > -lead_minutes))
    dep = keep(_departed_window(departure_events))
    targets = _materialized_join(
        targets, dep, (targets["Origin"] == dep["_airport"]) & (targets["_t_utc"] == dep["_t_utc"]),
        [dep["_dep_n"], dep["_dep_sum"], dep["_dep15"]], "origin departed 3h", keep, release)
    backlog = keep(_origin_backlog(events))
    targets = _materialized_join(
        targets, backlog,
        (targets["Origin"] == backlog["_airport"]) & (targets["_t_utc"] == backlog["_t_utc"]),
        [backlog["_sched_n"], backlog["_backlog_n"]], "origin backlog 3h", keep, release)

    arrival_events = events.filter(F.col("_actual_arr_utc_hour").isNull() |
                                   (F.col("_actual_arr_utc_hour") >= F.col("_sched_dep_utc_hour")))
    arrivals = keep(_landed_window(arrival_events))
    for airport_column, prefix in (("Origin", "_origin_arr"), ("Dest", "_dest_arr")):
        renamed = keep(arrivals.select(
            F.col("_airport").alias(f"{prefix}_airport"), F.col("_t_utc").alias(f"{prefix}_t_utc"),
            F.col("_arr_n").alias(f"{prefix}_n"), F.col("_arr_sum").alias(f"{prefix}_sum"),
            F.col("_arr15").alias(f"{prefix}_15")))
        targets = _materialized_join(
            targets, renamed,
            (targets[airport_column] == renamed[f"{prefix}_airport"]) &
            (targets["_t_utc"] == renamed[f"{prefix}_t_utc"]),
            [renamed[f"{prefix}_n"], renamed[f"{prefix}_sum"], renamed[f"{prefix}_15"]],
            f"{airport_column.lower()} landed 3h", keep, release)
    release(arrivals)

    carrier_dep = keep(_carrier_departed(departure_events))
    targets = _materialized_join(
        targets, carrier_dep,
        (targets["Reporting_Airline"] == carrier_dep["Reporting_Airline"]) &
        (targets["_t_day"] == carrier_dep["_t_day"]) & (targets["_t_hod"] == carrier_dep["_t_hod"]),
        [carrier_dep["_carrier_dep_n"], carrier_dep["_carrier_dep_sum"], carrier_dep["_carrier_dep15"]],
        "carrier departed UTC today", keep, release)
    carrier_backlog = keep(_carrier_backlog(events))
    targets = _materialized_join(
        targets, carrier_backlog,
        (targets["Reporting_Airline"] == carrier_backlog["Reporting_Airline"]) &
        (targets["_t_day"] == carrier_backlog["_t_day"]) &
        (targets["_t_hod"] == carrier_backlog["_t_hod"]),
        [carrier_backlog["_carrier_sched_n"], carrier_backlog["_carrier_backlog_n"]],
        "carrier backlog UTC today", keep, release)

    # The wide date/hour offset join plus six aggregates creates a large
    # auditable logical plan.  Reliably checkpoint after all row-count checks
    # so downstream smoothing/model plans fit the 1536 MB driver.
    if targets.sparkSession.sparkContext.getCheckpointDir():
        checkpointed_targets = keep(targets.checkpoint(eager=True))
    else:
        checkpointed_targets = keep(targets.localCheckpoint(eager=True))
    checkpointed_targets.count()
    release(targets)
    targets = checkpointed_targets
    print("ASOF_UTC_LINEAGE_CHECKPOINT=PASSED", flush=True)

    if priors is None:
        totals = targets.filter(F.col("FlightDate") < F.lit(prior_cutoff)).agg(
            F.sum(F.coalesce("_backlog_n", F.lit(0))).alias("b"),
            F.sum(F.coalesce("_sched_n", F.lit(0))).alias("n"),
            F.sum(F.coalesce("_carrier_backlog_n", F.lit(0))).alias("cb"),
            F.sum(F.coalesce("_carrier_sched_n", F.lit(0))).alias("cn")).first()
        fitted_priors["origin_backlog_share"] = float(totals["b"] or 0) / max(float(totals["n"] or 0), 1)
        fitted_priors["carrier_backlog_share"] = float(totals["cb"] or 0) / max(float(totals["cn"] or 0), 1)
        fitted_priors["smoothing"] = SMOOTHING

    def smooth_sum(sum_column, count_column, prior):
        return ((F.coalesce(F.col(sum_column), F.lit(0.0)) + F.lit(SMOOTHING * prior)) /
                (F.coalesce(F.col(count_column), F.lit(0.0)) + F.lit(SMOOTHING)))

    def smooth_share(positive_column, count_column, prior):
        return ((F.coalesce(F.col(positive_column), F.lit(0.0)) + F.lit(SMOOTHING * prior)) /
                (F.coalesce(F.col(count_column), F.lit(0.0)) + F.lit(SMOOTHING)))

    features = keep(targets.select(
        *JOIN_KEYS,
        F.coalesce("_dep_n", F.lit(0)).cast("double").alias("OpsOriginDepartedCount3h"),
        smooth_sum("_dep_sum", "_dep_n", fitted_priors["dep_mean"]).alias("OpsOriginMeanDepDelay3h"),
        smooth_share("_dep15", "_dep_n", fitted_priors["dep_share"]).alias("OpsOriginDepDelay15Share3h"),
        F.coalesce("_backlog_n", F.lit(0)).cast("double").alias("OpsOriginBacklogCount3h"),
        smooth_share("_backlog_n", "_sched_n", fitted_priors["origin_backlog_share"]).alias("OpsOriginBacklogShare3h"),
        F.coalesce("_origin_arr_n", F.lit(0)).cast("double").alias("OpsOriginInboundLandedCount3h"),
        smooth_sum("_origin_arr_sum", "_origin_arr_n", fitted_priors["arr_mean"]).alias("OpsOriginInboundMeanArrDelay3h"),
        smooth_share("_origin_arr_15", "_origin_arr_n", fitted_priors["arr_share"]).alias("OpsOriginInboundArrDelay15Share3h"),
        F.coalesce("_carrier_dep_n", F.lit(0)).cast("double").alias("OpsCarrierDepartedCountToday"),
        smooth_sum("_carrier_dep_sum", "_carrier_dep_n", fitted_priors["dep_mean"]).alias("OpsCarrierMeanDepDelayToday"),
        smooth_share("_carrier_dep15", "_carrier_dep_n", fitted_priors["dep_share"]).alias("OpsCarrierDepDelay15ShareToday"),
        F.coalesce("_carrier_sched_n", F.lit(0)).cast("double").alias("OpsCarrierScheduledBeforeCountToday"),
        smooth_share("_carrier_backlog_n", "_carrier_sched_n", fitted_priors["carrier_backlog_share"]).alias("OpsCarrierBacklogShareToday"),
        F.coalesce("_dest_arr_n", F.lit(0)).cast("double").alias("OpsDestLandedCount3h"),
        smooth_sum("_dest_arr_sum", "_dest_arr_n", fitted_priors["arr_mean"]).alias("OpsDestMeanArrDelay3h"),
        smooth_share("_dest_arr_15", "_dest_arr_n", fitted_priors["arr_share"]).alias("OpsDestArrDelay15Share3h")))
    features.count()
    expected = eligible.count()
    result = keep(eligible.join(features, JOIN_KEYS, "left"))
    actual = result.count()
    if actual != expected:
        raise ValueError(f"eligible UTC asof join changed rows {expected:,} -> {actual:,}")
    nulls = result.agg(*[F.sum(F.col(column).isNull().cast("long")).alias(column)
                         for column in OPS_FEATURES]).first().asDict()
    if any(int(value or 0) for value in nulls.values()):
        raise ValueError(f"UTC asof features contain nulls: {nulls}")
    print(f"ASOF_UTC_ROW_COUNT_CHECK=PASSED eligible feature join rows={actual}", flush=True)
    release(features); release(targets); release(events)
    stats = {"event_rows": expected_events, "event_timezone_matched": matched_events,
             "target_groups": target_rows, "target_timezone_matched": target_matched}
    return result, fitted_priors, stats
