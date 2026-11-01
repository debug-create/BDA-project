"""Hourly UTC operational state used by the host boosted/deep-model extension.

All steps are complete UTC hours strictly before the T-120 prediction cutoff.
The conversion and conservative DST rules are imported from ``asof_ops_utc``.
"""

from pyspark.sql import functions as F

import asof_ops_utc as utc


LAGS = tuple(range(1, 7))
STREAM_CHANNELS = {
    "origin_departures": ("count", "mean_delay", "delay15_share", "backlog_count", "mask"),
    "origin_inbound": ("count", "mean_delay", "delay15_share", "mask"),
    "carrier_network": ("count", "mean_delay", "delay15_share", "backlog_share", "mask"),
    "destination_arrivals": ("count", "mean_delay", "delay15_share", "mask"),
}


def _target_hours(eligible, timezone_table, lead_minutes):
    day = F.datediff("FlightDate", F.lit("1970-01-01").cast("date"))
    hour = F.col("ScheduledDepHour").cast("int")
    joined = eligible.alias("f").join(
        F.broadcast(timezone_table.alias("tz")),
        (F.col("f.Origin") == F.col("tz.Airport")) &
        (F.col("f.FlightDate") == F.col("tz.LocalDate")), "left")
    offset = F.greatest(utc._offset_for_hour("OffsetFold0", hour),
                        utc._offset_for_hour("OffsetFold1", hour))
    t_hour = F.floor((day * 1440 + hour * 60 - offset - lead_minutes) / 60).cast("long")
    return joined.select(
        *[F.col(f"f.{column}").alias(column) for column in eligible.columns],
        t_hour.alias("_state_t_hour"),
        F.floor(t_hour / 24).cast("long").alias("_state_t_day"),
        F.pmod(t_hour, F.lit(24)).cast("int").alias("_state_t_hod"),
        offset.isNotNull().cast("int").alias("StateTimezoneMatched"))


def _delay_hourly(events, airport_column, time_column, delay_column, prefix):
    valid = events.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
        F.col(delay_column).isNotNull() & F.col(time_column).isNotNull())
    return valid.groupBy(
        F.col(airport_column).alias(f"_{prefix}_airport"),
        F.col(time_column).alias(f"_{prefix}_hour")).agg(
        F.count("*").alias(f"_{prefix}_n"),
        F.sum(delay_column).alias(f"_{prefix}_sum"),
        F.sum((F.col(delay_column) >= 15).cast("long")).alias(f"_{prefix}_15"))


def _origin_backlog_snapshot(events):
    """Flights scheduled in the preceding three hours and still backlogged."""
    expanded = events.filter(F.col("_sched_dep_utc_hour").isNotNull()).select(
        F.col("Origin").alias("_ob_airport"), "Cancelled", "Diverted", "_actual_dep_utc_hour",
        F.explode(F.sequence(F.col("_sched_dep_utc_hour") + 1,
                             F.col("_sched_dep_utc_hour") + 3)).alias("_ob_hour"))
    backlogged = ((F.col("Cancelled") == 1) | (F.col("Diverted") == 1) |
                  F.col("_actual_dep_utc_hour").isNull() |
                  (F.col("_actual_dep_utc_hour") >= F.col("_ob_hour")))
    return expanded.groupBy("_ob_airport", "_ob_hour").agg(
        F.count("*").alias("_ob_sched"), F.sum(backlogged.cast("long")).alias("_ob_n"))


def _carrier_hourly(events):
    valid = events.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
        F.col("DepDelay").isNotNull() & F.col("_actual_dep_utc_hour").isNotNull())
    return valid.groupBy("Reporting_Airline", F.col("_actual_dep_utc_hour").alias("_cd_hour")).agg(
        F.count("*").alias("_cd_n"), F.sum("DepDelay").alias("_cd_sum"),
        F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_cd_15"))


def _route_earlier_today(events):
    valid = events.filter(
        (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) &
        F.col("DepDelay").isNotNull() & F.col("_actual_dep_utc_hour").isNotNull())
    day = F.floor(F.col("_actual_dep_utc_hour") / 24).cast("long")
    hod = F.pmod(F.col("_actual_dep_utc_hour"), F.lit(24)).cast("int")
    hourly = valid.groupBy("Origin", "Dest", day.alias("_route_day"), hod.alias("_event_hod")).agg(
        F.count("*").alias("_route_n0"), F.sum("DepDelay").alias("_route_sum0"),
        F.sum((F.col("DepDelay") >= 15).cast("long")).alias("_route_150"))
    expanded = hourly.filter(F.col("_event_hod") < 23).withColumn(
        "_route_hod", F.explode(F.sequence(F.col("_event_hod") + 1, F.lit(23))))
    return expanded.groupBy("Origin", "Dest", "_route_day", "_route_hod").agg(
        F.sum("_route_n0").alias("_route_n"), F.sum("_route_sum0").alias("_route_sum"),
        F.sum("_route_150").alias("_route_15"))


def attach_hourly_state_utc(event_source, eligible, timezone_table, lead_minutes,
                            priors, keep, release):
    """Attach six complete pre-T hourly steps and route-earlier-today features."""
    if "FlightKey" not in eligible.columns:
        raise ValueError("eligible frame must contain unique FlightKey")
    expected = eligible.count()
    events = keep(utc.add_utc_event_times(event_source, timezone_table).select(
        "Reporting_Airline", "Origin", "Dest", "Cancelled", "Diverted", "DepDelay", "ArrDelay",
        "_sched_dep_utc_hour", "_actual_dep_utc_hour", "_actual_arr_utc_hour"))

    targets = keep(_target_hours(eligible, timezone_table, lead_minutes))
    steps = keep(targets.select(
        "FlightKey", "Origin", "Dest", "Reporting_Airline", "_state_t_hour",
        F.explode(F.array(*[F.lit(lag) for lag in LAGS])).alias("LagHour"))
        .withColumn("_snapshot_hour", F.col("_state_t_hour") - F.col("LagHour")))

    dep_events = events.filter(F.col("DepDelay").isNull() | (F.col("DepDelay") > -lead_minutes))
    arrivals_events = events.filter(F.col("_actual_arr_utc_hour").isNull() |
                                    (F.col("_actual_arr_utc_hour") >= F.col("_sched_dep_utc_hour")))
    od = keep(_delay_hourly(dep_events, "Origin", "_actual_dep_utc_hour", "DepDelay", "od"))
    arr = keep(_delay_hourly(arrivals_events, "Dest", "_actual_arr_utc_hour", "ArrDelay", "arr"))
    ob = keep(_origin_backlog_snapshot(events))
    cd = keep(_carrier_hourly(dep_events))
    cb = keep(utc._carrier_backlog(events))

    x = steps.alias("s").join(
        od.alias("od"), (F.col("s.Origin") == F.col("od._od_airport")) &
        (F.col("s._snapshot_hour") == F.col("od._od_hour")), "left")
    x = x.join(ob.alias("ob"), (F.col("s.Origin") == F.col("ob._ob_airport")) &
               (F.col("s._snapshot_hour") == F.col("ob._ob_hour")), "left")
    x = x.join(arr.alias("oi"), (F.col("s.Origin") == F.col("oi._arr_airport")) &
               (F.col("s._snapshot_hour") == F.col("oi._arr_hour")), "left")
    x = x.join(arr.alias("da"), (F.col("s.Dest") == F.col("da._arr_airport")) &
               (F.col("s._snapshot_hour") == F.col("da._arr_hour")), "left")
    x = x.join(cd.alias("cd"), (F.col("s.Reporting_Airline") == F.col("cd.Reporting_Airline")) &
               (F.col("s._snapshot_hour") == F.col("cd._cd_hour")), "left")
    snap_day = F.floor(F.col("s._snapshot_hour") / 24).cast("long")
    snap_hod = F.pmod(F.col("s._snapshot_hour"), F.lit(24)).cast("int")
    x = x.join(cb.alias("cb"), (F.col("s.Reporting_Airline") == F.col("cb.Reporting_Airline")) &
               (snap_day == F.col("cb._t_day")) & (snap_hod == F.col("cb._t_hod")), "left")

    smooth = float(priors.get("smoothing", 20.0))
    def mean(sum_col, n_col, prior):
        return ((F.coalesce(F.col(sum_col), F.lit(0.0)) + F.lit(smooth * float(prior))) /
                (F.coalesce(F.col(n_col), F.lit(0.0)) + F.lit(smooth)))
    def share(pos_col, n_col, prior):
        return mean(pos_col, n_col, prior)

    step_values = x.select(
        F.col("s.FlightKey").alias("FlightKey"), F.col("s.LagHour").alias("LagHour"),
        F.log1p(F.coalesce(F.col("od._od_n"), F.lit(0))).alias("origin_departures_count"),
        mean("od._od_sum", "od._od_n", priors["dep_mean"]).alias("origin_departures_mean_delay"),
        share("od._od_15", "od._od_n", priors["dep_share"]).alias("origin_departures_delay15_share"),
        F.log1p(F.coalesce(F.col("ob._ob_n"), F.lit(0))).alias("origin_departures_backlog_count"),
        (F.coalesce(F.col("od._od_n"), F.lit(0)) > 0).cast("double").alias("origin_departures_mask"),
        F.log1p(F.coalesce(F.col("oi._arr_n"), F.lit(0))).alias("origin_inbound_count"),
        mean("oi._arr_sum", "oi._arr_n", priors["arr_mean"]).alias("origin_inbound_mean_delay"),
        share("oi._arr_15", "oi._arr_n", priors["arr_share"]).alias("origin_inbound_delay15_share"),
        (F.coalesce(F.col("oi._arr_n"), F.lit(0)) > 0).cast("double").alias("origin_inbound_mask"),
        F.log1p(F.coalesce(F.col("cd._cd_n"), F.lit(0))).alias("carrier_network_count"),
        mean("cd._cd_sum", "cd._cd_n", priors["dep_mean"]).alias("carrier_network_mean_delay"),
        share("cd._cd_15", "cd._cd_n", priors["dep_share"]).alias("carrier_network_delay15_share"),
        share("cb._carrier_backlog_n", "cb._carrier_sched_n",
              priors["carrier_backlog_share"]).alias("carrier_network_backlog_share"),
        (F.coalesce(F.col("cd._cd_n"), F.lit(0)) > 0).cast("double").alias("carrier_network_mask"),
        F.log1p(F.coalesce(F.col("da._arr_n"), F.lit(0))).alias("destination_arrivals_count"),
        mean("da._arr_sum", "da._arr_n", priors["arr_mean"]).alias("destination_arrivals_mean_delay"),
        share("da._arr_15", "da._arr_n", priors["arr_share"]).alias("destination_arrivals_delay15_share"),
        (F.coalesce(F.col("da._arr_n"), F.lit(0)) > 0).cast("double").alias("destination_arrivals_mask"))

    value_names = [f"{stream}_{channel}" for stream, channels in STREAM_CHANNELS.items()
                   for channel in channels]
    expressions = [
        F.max(F.when(F.col("LagHour") == lag, F.col(name))).alias(f"seq_{name}_h{lag}")
        for name in value_names for lag in LAGS]
    wide = keep(step_values.groupBy("FlightKey").agg(*expressions))

    routes = keep(_route_earlier_today(dep_events))
    route_features = targets.alias("t").join(
        routes.alias("r"),
        (F.col("t.Origin") == F.col("r.Origin")) & (F.col("t.Dest") == F.col("r.Dest")) &
        (F.col("t._state_t_day") == F.col("r._route_day")) &
        (F.col("t._state_t_hod") == F.col("r._route_hod")), "left").select(
        F.col("t.FlightKey").alias("FlightKey"),
        F.log1p(F.coalesce(F.col("r._route_n"), F.lit(0))).alias("RouteEarlierTodayCountLog"),
        mean("r._route_sum", "r._route_n", priors["dep_mean"]).alias("RouteEarlierTodayMeanDepDelay"),
        share("r._route_15", "r._route_n", priors["dep_share"]).alias("RouteEarlierTodayDepDelay15Share"),
        (F.coalesce(F.col("r._route_n"), F.lit(0)) > 0).cast("double").alias("RouteEarlierTodayMask"))

    result = keep(targets.join(wide, "FlightKey", "left").join(route_features, "FlightKey", "left"))
    actual = result.count()
    if actual != expected:
        raise ValueError(f"hourly state joins changed rows {expected:,} -> {actual:,}")
    state_columns = [column for column in result.columns if column.startswith("seq_")]
    nulls = result.agg(*[F.sum(F.col(column).isNull().cast("long")).alias(column)
                         for column in state_columns]).first().asDict()
    if any(int(value or 0) for value in nulls.values()):
        raise ValueError("hourly state contains nulls")
    print(f"ASOF_STATE_ROW_COUNT_CHECK=PASSED rows={actual} columns={len(state_columns)}", flush=True)
    for frame in (wide, routes, targets, steps, od, arr, ob, cd, cb, events):
        release(frame)
    return result, state_columns
