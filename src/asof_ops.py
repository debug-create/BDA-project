"""Conservative hourly operations-aware features for flight-delay prediction.

BTS scheduled times are local to each airport.  No UTC conversion is attempted:
departure events use origin-local clocks and arrival events use destination-local
clocks.  Scheduled arrival is assigned to the following local calendar day when
CRSArrTime < CRSDepTime, then ArrDelay is added.  This is the documented arrival
date approximation available from the clean handoff.
"""

from pyspark.sql import functions as F


SMOOTHING = 20.0
JOIN_KEYS = ["FlightDate", "Origin", "Dest", "Reporting_Airline", "ScheduledDepHour"]
REQUIRED_EVENT_COLUMNS = ["DepDelay", "ArrDelay", "Cancelled", "Diverted",
                          "CRSDepTime", "CRSArrTime", "FlightDate",
                          "Reporting_Airline", "Origin", "Dest"]
OPS_FEATURES = [
    "OpsOriginDepartedCount3h", "OpsOriginMeanDepDelay3h", "OpsOriginDepDelay15Share3h",
    "OpsOriginBacklogCount3h", "OpsOriginBacklogShare3h",
    "OpsOriginInboundLandedCount3h", "OpsOriginInboundMeanArrDelay3h", "OpsOriginInboundArrDelay15Share3h",
    "OpsCarrierDepartedCountToday", "OpsCarrierMeanDepDelayToday", "OpsCarrierDepDelay15ShareToday",
    "OpsCarrierScheduledBeforeCountToday", "OpsCarrierBacklogShareToday",
    "OpsDestLandedCount3h", "OpsDestMeanArrDelay3h", "OpsDestArrDelay15Share3h",
]


def minute_of_day(column):
    value = column.cast("int")
    hour = F.floor(value / 100).cast("int")
    minute = (value % 100).cast("int")
    return F.when(value == 2400, F.lit(0)).when(
        value.isNotNull() & hour.between(0, 23) & minute.between(0, 59), hour * 60 + minute)


def _empty_int_array():
    return F.array().cast("array<int>")


def _sequence_if(start, end, condition):
    return F.when(condition, F.sequence(start.cast("int"), end.cast("int"))).otherwise(_empty_int_array())


def add_event_time_columns(flights):
    dep_min = minute_of_day(F.col("CRSDepTime"))
    arr_min = minute_of_day(F.col("CRSArrTime"))
    day = F.datediff("FlightDate", F.lit("1970-01-01").cast("date"))
    scheduled_departure = day * 1440 + dep_min
    scheduled_arrival = (day + F.when(arr_min < dep_min, 1).otherwise(0)) * 1440 + arr_min
    return (flights
            .withColumn("_sched_dep_abs_hour", F.floor(scheduled_departure / 60).cast("long"))
            .withColumn("_actual_dep_abs_hour",
                        F.floor((scheduled_departure + F.col("DepDelay")) / 60).cast("long"))
            .withColumn("_actual_arr_abs_hour",
                        F.floor((scheduled_arrival + F.col("ArrDelay")) / 60).cast("long")))


def _outcome_priors(events, cutoff):
    train = events.filter(F.col("FlightDate") < F.lit(cutoff))
    departures = train.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
                              F.col("DepDelay").isNotNull())
    arrivals = train.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
                            F.col("ArrDelay").isNotNull())
    dep = departures.agg(F.avg("DepDelay").alias("mean"),
                         F.avg((F.col("DepDelay") >= 15).cast("double")).alias("share")).first()
    arr = arrivals.agg(F.avg("ArrDelay").alias("mean"),
                       F.avg((F.col("ArrDelay") >= 15).cast("double")).alias("share")).first()
    return {"dep_mean": float(dep["mean"]), "dep_share": float(dep["share"]),
            "arr_mean": float(arr["mean"]), "arr_share": float(arr["share"])}


def _departed_window(events):
    valid = events.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
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


def _landed_window(events):
    valid = events.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
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


def _origin_backlog(events):
    expanded = events.select(
        F.col("Origin").alias("_airport"), "Cancelled", "Diverted", "_actual_dep_abs_hour",
        F.explode(F.sequence(F.col("_sched_dep_abs_hour") + 1,
                             F.col("_sched_dep_abs_hour") + 3)).alias("_t_abs"))
    backlog = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
               (F.col("_actual_dep_abs_hour") >= F.col("_t_abs")))
    return (expanded.groupBy("_airport", "_t_abs")
            .agg(F.count("*").alias("_sched_n"), F.sum(backlog.cast("long")).alias("_backlog_n")))


def _carrier_departed(events):
    valid = events.filter((F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
                          F.col("DepDelay").isNotNull() & F.col("_actual_dep_abs_hour").isNotNull())
    event_day = F.floor(F.col("_actual_dep_abs_hour") / 24).cast("long")
    event_hour = F.pmod(F.col("_actual_dep_abs_hour"), F.lit(24)).cast("int")
    hourly = (valid.groupBy("Reporting_Airline", event_day.alias("_t_day"), event_hour.alias("_event_hod"))
              .agg(F.count("*").alias("_dep_n"), F.sum("DepDelay").alias("_dep_sum"),
                   F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_dep15")))
    expanded = hourly.withColumn(
        "_t_hod", F.explode(_sequence_if(F.col("_event_hod") + 1, F.lit(23), F.col("_event_hod") < 23)))
    return (expanded.groupBy("Reporting_Airline", "_t_day", "_t_hod")
            .agg(F.sum("_dep_n").alias("_carrier_dep_n"), F.sum("_dep_sum").alias("_carrier_dep_sum"),
                 F.sum("_dep15").alias("_carrier_dep15")))


def _carrier_backlog(events):
    sched_day = F.floor(F.col("_sched_dep_abs_hour") / 24).cast("long")
    sched_hour = F.pmod(F.col("_sched_dep_abs_hour"), F.lit(24)).cast("int")
    base = events.select("Reporting_Airline", "Cancelled", "Diverted", "_actual_dep_abs_hour",
                         sched_day.alias("_t_day"), sched_hour.alias("_sched_hod"))
    expanded = base.withColumn(
        "_t_hod", F.explode(_sequence_if(F.col("_sched_hod") + 1, F.lit(23), F.col("_sched_hod") < 23)))
    snapshot = F.col("_t_day") * 24 + F.col("_t_hod")
    backlog = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
               (F.col("_actual_dep_abs_hour") >= snapshot))
    return (expanded.groupBy("Reporting_Airline", "_t_day", "_t_hod")
            .agg(F.count("*").alias("_carrier_sched_n"),
                 F.sum(backlog.cast("long")).alias("_carrier_backlog_n")))


def _materialized_join(targets, block, condition, output_columns, label, keep, release):
    expected = targets.count()
    joined = targets.join(block, condition, "left")
    actual = joined.count()
    if actual != expected:
        raise ValueError(f"{label}: row count changed from {expected:,} to {actual:,}")
    print(f"ASOF_ROW_COUNT_CHECK=PASSED {label} rows={actual}", flush=True)
    next_targets = keep(joined.select(*[targets[column] for column in targets.columns], *output_columns))
    next_targets.count()
    release(targets)
    release(block)
    return next_targets


def attach_asof_features(event_source, eligible, lead_minutes, prior_cutoff, keep, release, priors=None):
    """Return eligible rows plus as-of features and train-fitted smoothing priors."""
    if lead_minutes not in (60, 120, 180):
        raise ValueError("lead_minutes must be 60, 120, or 180")
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in event_source.columns]
    if missing:
        raise ValueError(f"asof_ops requires clean columns: {missing}")
    events = keep(add_event_time_columns(event_source).select(
        "FlightDate", "Reporting_Airline", "Origin", "Dest", "Cancelled", "Diverted",
        "DepDelay", "ArrDelay", "_sched_dep_abs_hour", "_actual_dep_abs_hour", "_actual_arr_abs_hour"))
    events.count()
    fitted_priors = dict(priors) if priors is not None else _outcome_priors(events, prior_cutoff)

    targets = keep(eligible.select(*JOIN_KEYS).distinct()
                   .withColumn("_flight_day", F.datediff("FlightDate", F.lit("1970-01-01").cast("date")))
                   .withColumn("_t_abs", F.col("_flight_day") * 24 + F.col("ScheduledDepHour").cast("long") - lead_minutes // 60)
                   .withColumn("_t_day", F.floor(F.col("_t_abs") / 24).cast("long"))
                   .withColumn("_t_hod", F.pmod(F.col("_t_abs"), F.lit(24)).cast("int")))
    targets.count()

    # A target flight can never contribute its own departure outcome.  With no
    # source flight identifier in the clean handoff, conservatively exclude the
    # anomalous class of events departing at least the full lead time early;
    # otherwise such a row could appear before its own T hour.
    departure_events = events.filter(F.col("DepDelay").isNull() | (F.col("DepDelay") > -lead_minutes))
    dep = keep(_departed_window(departure_events))
    targets = _materialized_join(
        targets, dep, (targets["Origin"] == dep["_airport"]) & (targets["_t_abs"] == dep["_t_abs"]),
        [dep["_dep_n"], dep["_dep_sum"], dep["_dep15"]], "origin departed 3h", keep, release)
    backlog = keep(_origin_backlog(events))
    targets = _materialized_join(
        targets, backlog,
        (targets["Origin"] == backlog["_airport"]) & (targets["_t_abs"] == backlog["_t_abs"]),
        [backlog["_sched_n"], backlog["_backlog_n"]], "origin backlog 3h", keep, release)

    # Likewise discard impossible/anomalous arrivals before their own scheduled
    # departure hour, which guarantees the target cannot supply its own arrival.
    arrival_events = events.filter(F.col("_actual_arr_abs_hour").isNull() |
                                   (F.col("_actual_arr_abs_hour") >= F.col("_sched_dep_abs_hour")))
    arrivals = keep(_landed_window(arrival_events))
    for airport_column, prefix in (("Origin", "_origin_arr"), ("Dest", "_dest_arr")):
        renamed = keep(arrivals.select(
            F.col("_airport").alias(f"{prefix}_airport"), F.col("_t_abs").alias(f"{prefix}_t_abs"),
            F.col("_arr_n").alias(f"{prefix}_n"), F.col("_arr_sum").alias(f"{prefix}_sum"),
            F.col("_arr15").alias(f"{prefix}_15")))
        targets = _materialized_join(
            targets, renamed,
            (targets[airport_column] == renamed[f"{prefix}_airport"]) &
            (targets["_t_abs"] == renamed[f"{prefix}_t_abs"]),
            [renamed[f"{prefix}_n"], renamed[f"{prefix}_sum"], renamed[f"{prefix}_15"]],
            f"{airport_column.lower()} landed 3h", keep, release)
    release(arrivals)

    carrier_dep = keep(_carrier_departed(departure_events))
    expected = targets.count()
    joined = targets.join(carrier_dep, ["Reporting_Airline", "_t_day", "_t_hod"], "left")
    actual = joined.count()
    if actual != expected:
        raise ValueError(f"carrier departed today: row count changed from {expected:,} to {actual:,}")
    print(f"ASOF_ROW_COUNT_CHECK=PASSED carrier departed today rows={actual}", flush=True)
    next_targets = keep(joined.select(*targets.columns, "_carrier_dep_n", "_carrier_dep_sum", "_carrier_dep15"))
    next_targets.count(); release(targets); release(carrier_dep); targets = next_targets

    carrier_backlog = keep(_carrier_backlog(events))
    expected = targets.count()
    joined = targets.join(carrier_backlog, ["Reporting_Airline", "_t_day", "_t_hod"], "left")
    actual = joined.count()
    if actual != expected:
        raise ValueError(f"carrier backlog today: row count changed from {expected:,} to {actual:,}")
    print(f"ASOF_ROW_COUNT_CHECK=PASSED carrier backlog today rows={actual}", flush=True)
    next_targets = keep(joined.select(*targets.columns, "_carrier_sched_n", "_carrier_backlog_n"))
    next_targets.count(); release(targets); release(carrier_backlog); targets = next_targets

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

    features = targets.select(
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
        smooth_share("_dest_arr_15", "_dest_arr_n", fitted_priors["arr_share"]).alias("OpsDestArrDelay15Share3h"))
    features = keep(features); features.count()
    expected = eligible.count()
    result = keep(eligible.join(features, JOIN_KEYS, "left"))
    actual = result.count()
    if actual != expected:
        raise ValueError(f"eligible asof_ops join: row count changed from {expected:,} to {actual:,}")
    nulls = result.agg(*[F.sum(F.col(column).isNull().cast("long")).alias(column)
                         for column in OPS_FEATURES]).first().asDict()
    if any(int(value or 0) for value in nulls.values()):
        raise ValueError(f"asof_ops produced null features: {nulls}")
    print(f"ASOF_ROW_COUNT_CHECK=PASSED eligible feature join rows={actual}", flush=True)
    release(features); release(targets); release(events)
    return result, fitted_priors
