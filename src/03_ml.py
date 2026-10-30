#!/usr/bin/env python3
"""Spark MLlib classification of BTS flight arrival delays (ArrDel15) -- redesigned ML stage.

Question: using ONLY information available before scheduled departure, can we predict
whether an eligible scheduled flight arrives >= 15 minutes late (ArrDel15 = 1)?
This is an observational prediction task on three months of data. It is NOT a causal
estimate of any holiday effect (the descriptive Spark SQL job is the place for that).

What is fixed (agreed with the project owner, unchanged here)
-------------------------------------------------------------
* Target ArrDel15; cancelled / diverted flights and rows without a 0/1 target excluded.
* Chronological split by whole FlightDate: 60% / 10% / 30% of the distinct dates
  (train 2024-10-01..11-24, validation 11-25..12-03, test 12-04..12-31 on the real data).
* Logistic Regression and Random Forest are both always trained, tuned and reported.
* Hyperparameters and thresholds are chosen on VALIDATION data only (AUC-PR is the
  primary ranking metric because the classes are imbalanced; the threshold maximises
  validation F1 of the delayed class).

What this script adds (see docs/ML_REDESIGN_AUDIT_AND_EXPERIMENT_PLAN.md)
------------------------------------------------------------------------
1. Output namespaces. Default is an *experiments* namespace so the baseline results of
   the production run are never overwritten by accident. Writing to the production
   paths needs both --namespace production and --confirm-overwrite-baseline.
2. Two run modes. --mode select reads ONLY train + validation rows (test rows are
   excluded from all label-based selection operations), so test labels cannot influence any
   decision during development. --mode all does the same selection, writes the frozen
   decisions to frozen_config.json, and only then refits and scores the test window once.
3. A compact one-factor-at-a-time feature experiment table (validation only), each
   compared with the reference design (the previous script's feature set) using LR:
     E0 reference | E1 no history | E2 no holiday features | E3 history warm-up |
     E4 trailing-window history | E5 + scheduled airport-hour demand
   A challenger replaces the reference only if its validation AUC-PR is better by at
   least SELECTION_MARGIN (a deliberate guard against chasing noise).
4. Runtime checks: eligibility categories reconcile with the input row count, every
   join is verified not to change the row count, and the target must be binary.
5. Threshold-transfer diagnostics (validation vs test predicted-positive rate) because a
   threshold picked on one period is applied to another period with a different
   delay rate.

Feature rules (all known before departure)
------------------------------------------
* scheduled departure/arrival time parsed from HHMM into minute-of-day (2400 -> 0),
  then cyclic sin/cos plus scheduled hour; scheduled elapsed time; distance;
* carrier, day of week (one-hot), holiday period and holiday flag (calendar file);
* smoothed historical delay rates for carrier / origin / destination / route /
  origin x departure-block. For a FITTING row on date D the rates use labels from
  dates strictly before D only. Rows in a later window (validation, test) use rates
  from the completed fitting window only; their own labels are never used;
* optional scheduled demand: number of flights scheduled from the same origin, on the
  same date, in the same scheduled departure hour. It is counted over ALL scheduled
  flights including cancelled/diverted ones, because dropping those would leak
  outcome information (a cancellation is only known after the fact).
Raw Year / Month / DayofMonth / DayOfYear are deliberately NOT used: in a chronological
split the test dates lie outside the training date range.

Limitations that code cannot fix: no weather, inbound-aircraft status or same-day
network state is allowed (not known before departure); delay prevalence drifts across
the three months; Thanksgiving / Christmas categories are rare or absent in the fitting
windows, so the model can only extrapolate holiday behaviour from other holidays.
Better metrics are not guaranteed; whatever the held-out test window gives is reported.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import (GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier,
                                        RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window, functions as F, types as T

try:
    from xgboost.spark import SparkXGBClassifier
except ImportError:
    # XGBoost is optional: ordinary LR/RF/GBT/MLP runs must still work in the
    # original image. A clear error is raised only when --extra-models xgb is used.
    SparkXGBClassifier = None

# --------------------------------------------------------------------------
# Paths. The production names are the existing interface and are unchanged.
# --------------------------------------------------------------------------
CLEAN_PATH = "hdfs://master:9000/user/root/flights/clean"
HOLIDAYS_PATH = "hdfs://master:9000/user/root/flights/reference/holidays.csv"
PRODUCTION_HDFS_RESULTS = "hdfs://master:9000/user/root/flights/results/ml"
PRODUCTION_HDFS_MODELS = "hdfs://master:9000/user/root/flights/models"
PRODUCTION_LOCAL_RESULTS = "/results/ml"
EXPERIMENT_HDFS_ROOT = "hdfs://master:9000/user/root/flights/experiments/ml_redesign"
EXPERIMENT_LOCAL_ROOT = "/results/experiments/ml_redesign"

# --------------------------------------------------------------------------
# Settings (edit here to make the job lighter or heavier)
# --------------------------------------------------------------------------
VALIDATION_START_DATE_FRACTION = 0.60   # unchanged: identical windows to the baseline run
TEST_START_DATE_FRACTION = 0.70
# Widened from 0.60 to 0.90 so that class-weighted models (whose probabilities are
# inflated) cannot hit the grid edge. The unweighted search range is a subset.
THRESHOLD_CANDIDATES = [round(0.02 + 0.01 * i, 2) for i in range(89)]   # 0.02 ... 0.90
TOP_FRACTIONS = [0.10, 0.20, 0.30]      # "flag the riskiest X% of flights" lift table
SEED = 42

SELECTION_MARGIN = 0.002                # AUC-PR gain a challenger needs over the reference
EXPERIMENT_LR_REG = 0.01                # fixed LR strength for the feature experiments

HISTORY_SMOOTHING = 100.0               # pseudo-flights pulling rates toward the prior
HISTORY_KEYS = {
    "hist_rate_airline": ["Reporting_Airline"],
    "hist_rate_origin": ["Origin"],
    "hist_rate_dest": ["Dest"],
    "hist_rate_route": ["Origin", "Dest"],
    "hist_rate_origin_depblk": ["Origin", "DepTimeBlk"],
}

# Additional past-only history keys (spec factor history_set="extended"). The first block of
# columns exists in the eligible frame; DayOfWeekCat / ScheduledDepHour / ScheduledArrHour are
# derived from schedule fields only.
EXTRA_HISTORY_KEYS = {
    "hist_rate_airline_origin": ["Reporting_Airline", "Origin"],
    "hist_rate_origin_dow": ["Origin", "DayOfWeekCat"],
    "hist_rate_airline_depblk": ["Reporting_Airline", "DepTimeBlk"],
    "hist_rate_dest_arrhour": ["Dest", "ScheduledArrHour"],
    "hist_rate_airline_route": ["Reporting_Airline", "Origin", "Dest"],
}
# Short-memory rates (spec factor recent_days): the same idea over only the last N dates, to
# follow a drifting delay rate. Fitted separately from the long-memory rates.
RECENT_HISTORY_KEYS = {
    "recent_rate_airline": ["Reporting_Airline"],
    "recent_rate_origin": ["Origin"],
    "recent_rate_origin_depblk": ["Origin", "DepTimeBlk"],
}
RECENT_DAYS_CHOICE = 7

LR_MAX_ITER = 30
LR_REG_GRID = [0.001, 0.01, 0.1]        # L2 strength, 3 candidates

RF_NUM_TREES = 30
RF_CONFIGS = [                          # 2 candidates, chosen on validation AUC-PR
    {"maxDepth": 5, "minInstancesPerNode": 100},
    {"maxDepth": 7, "minInstancesPerNode": 100},
]
RF_MAX_BINS = 32
RF_SUBSAMPLING_RATE = 0.5
RF_FEATURE_SUBSET = "onethird"
RF_MAX_MEMORY_MB = 128                  # small histogram budget for 512 MB executors

# Session settings applied when this script creates its own SparkSession. The pipeline runs
# thousands of small jobs; by default the driver keeps metadata (including query plans) for up
# to 1000 jobs/stages/SQL executions, which is real memory on a 512 MB driver. These cap it.
SPARK_SESSION_CONF = {
    "spark.ui.retainedJobs": "100",
    "spark.ui.retainedStages": "100",
    "spark.ui.retainedTasks": "1000",
    "spark.sql.ui.retainedExecutions": "50",
    # Repeated experiment feature plans contain many joins/windows. AQE's
    # onUpdatePlan path renders every rewritten physical plan for the SQL UI;
    # that plan string alone exhausted a 2 GB driver in the regression suite.
    # Static planning is deterministic here and avoids that driver-only cost.
    "spark.sql.adaptive.enabled": "false",
    "spark.sql.ui.explainMode": "simple",
    "spark.sql.debug.maxToStringFields": "50",
}

# Extra model families (opt in with --extra-models gbt,mlp,xgb). They are real candidates: tuned on
# validation AUC-PR, thresholded on validation F1, refit on train+validation, scored once on test.
# Runtime/memory on 1.7 M rows with 512 MB executors is UNTESTED; start with --mode select.
GBT_CONFIGS = [{"maxIter": 30, "maxDepth": 4}, {"maxIter": 30, "maxDepth": 6}]
GBT_STEP_SIZE = 0.1
GBT_MIN_INSTANCES = 100
MLP_CONFIGS = [{"hidden": [16]}, {"hidden": [32, 16]}]     # hidden layer sizes
MLP_MAX_ITER = 60
MLP_BLOCK_SIZE = 256
XGB_CONFIGS = [
    {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 20.0},
    {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 20.0},
]
XGB_NUM_WORKERS = 2
EXTRA_MODEL_NAMES = {
    "gbt": "GBTClassifier", "mlp": "MultilayerPerceptron", "xgb": "SparkXGBClassifier"
}
MODEL_DIRS = {"LogisticRegression": "logistic_regression", "RandomForest": "random_forest",
              "GBTClassifier": "gbt", "MultilayerPerceptron": "mlp",
              "SparkXGBClassifier": "xgboost"}

# --------------------------------------------------------------------------
# Feature definitions and leakage rules
# --------------------------------------------------------------------------
# Source columns the predictors are built from (all known before departure).
# FlightDate is used only as the calendar key (history ordering, demand grouping).
FEATURE_SOURCE_COLUMNS = [
    "FlightDate", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "Reporting_Airline",
    "DayOfWeek", "Origin", "Dest", "DepTimeBlk", "HolidayPeriod",
]
REQUIRED_CLEAN_COLUMNS = ["Cancelled", "Diverted", "ArrDel15"] + [
    c for c in FEATURE_SOURCE_COLUMNS if c != "HolidayPeriod"
]
# Expected Spark simpleString types of the required clean columns.
NUMERIC_TYPES = {"tinyint", "smallint", "int", "bigint", "float", "double"}
REQUIRED_COLUMN_KINDS = {
    "FlightDate": {"date"},
    "Cancelled": NUMERIC_TYPES, "Diverted": NUMERIC_TYPES, "ArrDel15": NUMERIC_TYPES,
    "CRSDepTime": NUMERIC_TYPES, "CRSArrTime": NUMERIC_TYPES, "CRSElapsedTime": NUMERIC_TYPES,
    "Distance": NUMERIC_TYPES, "DayOfWeek": NUMERIC_TYPES,
    "Reporting_Airline": {"string"}, "Origin": {"string"}, "Dest": {"string"}, "DepTimeBlk": {"string"},
}
FORBIDDEN_SOURCE_COLUMNS = {
    "ArrDel15", "ArrDelay", "ArrDelayMinutes", "ArrivalDelayGroups",
    "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups", "DepTime",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "ArrTime", "ActualElapsedTime",
    "AirTime", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "Cancelled", "CancellationCode", "Diverted", "Tail_Number",
}

BASE_NUMERIC_FEATURES = [
    "DepMinuteSin", "DepMinuteCos", "ArrMinuteSin", "ArrMinuteCos",
    "ScheduledDepHour", "ScheduledArrHour",
]
HOLIDAY_NUMERIC_FEATURES = ["IsHolidayWindow"]
DEMAND_FEATURES = ["OriginHourDemandLog"]
# Scheduled-congestion extras (spec factor extra_demand): all from counts of SCHEDULED flights.
EXTRA_DEMAND_FEATURES = ["DestHourDemandLog", "OriginDayDemandLog", "OriginPrevHourDemandLog"]
# Schedule-shape extras (spec factor sched_extras): block-time padding / scheduled speed.
SCHED_EXTRA_FEATURES = ["SchedSpeedMph", "DistanceLog"]
IMPUTE_FEATURES = ["CRSElapsedTime", "Distance"]
HISTORY_FEATURES = list(HISTORY_KEYS.keys())
BASE_CATEGORICAL_FEATURES = ["Reporting_Airline", "DayOfWeekCat"]
HOLIDAY_CATEGORICAL_FEATURES = ["HolidayPeriod"]

# Derived feature -> the source columns it is computed from. Used by the leakage
# guard and by the tests. History rates additionally use the label of EARLIER
# dates only (see apply_history_training_past_only / apply_history).
FEATURE_PROVENANCE = {
    "DepMinuteSin": ["CRSDepTime"], "DepMinuteCos": ["CRSDepTime"],
    "ArrMinuteSin": ["CRSArrTime"], "ArrMinuteCos": ["CRSArrTime"],
    "ScheduledDepHour": ["CRSDepTime"], "ScheduledArrHour": ["CRSArrTime"],
    "IsHolidayWindow": ["HolidayPeriod"], "HolidayPeriod": ["HolidayPeriod"],
    "CRSElapsedTime": ["CRSElapsedTime"], "Distance": ["Distance"],
    "Reporting_Airline": ["Reporting_Airline"], "DayOfWeekCat": ["DayOfWeek"],
    "OriginHourDemandLog": ["Origin", "FlightDate", "CRSDepTime"],
    "DestHourDemandLog": ["Dest", "FlightDate", "CRSArrTime"],
    "OriginDayDemandLog": ["Origin", "FlightDate"],
    "OriginPrevHourDemandLog": ["Origin", "FlightDate", "CRSDepTime"],
    "SchedSpeedMph": ["Distance", "CRSElapsedTime"], "DistanceLog": ["Distance"],
    "hist_rate_airline": ["Reporting_Airline", "FlightDate"],
    "hist_rate_origin": ["Origin", "FlightDate"],
    "hist_rate_dest": ["Dest", "FlightDate"],
    "hist_rate_route": ["Origin", "Dest", "FlightDate"],
    "hist_rate_origin_depblk": ["Origin", "DepTimeBlk", "FlightDate"],
    "hist_rate_airline_origin": ["Reporting_Airline", "Origin", "FlightDate"],
    "hist_rate_origin_dow": ["Origin", "DayOfWeek", "FlightDate"],
    "hist_rate_airline_depblk": ["Reporting_Airline", "DepTimeBlk", "FlightDate"],
    "hist_rate_dest_arrhour": ["Dest", "CRSArrTime", "FlightDate"],
    "hist_rate_airline_route": ["Reporting_Airline", "Origin", "Dest", "FlightDate"],
    "recent_rate_airline": ["Reporting_Airline", "FlightDate"],
    "recent_rate_origin": ["Origin", "FlightDate"],
    "recent_rate_origin_depblk": ["Origin", "DepTimeBlk", "FlightDate"],
}

# One-factor-at-a-time feature experiments. Each differs from E0 in ONE factor.
SPEC_DEFAULTS = {
    "use_history": True, "window_days": None, "warmup_days": 0,
    "use_holiday": True, "use_demand": False,
    "history_set": "base", "recent_days": None, "extra_demand": False, "sched_extras": False,
}
EXPERIMENT_SPECS = [
    {"name": "E0_reference", "factor": "none (previous script's feature set)", **SPEC_DEFAULTS},
    {"name": "E1_no_history", "factor": "remove historical-rate features", **{**SPEC_DEFAULTS, "use_history": False}},
    {"name": "E2_no_holiday_features", "factor": "remove holiday flag/category", **{**SPEC_DEFAULTS, "use_holiday": False}},
    {"name": "E3_history_warmup7", "factor": "drop first 7 fitting days as rows (still used as history)",
     **{**SPEC_DEFAULTS, "warmup_days": 7}},
    {"name": "E4_history_trailing14", "factor": "history uses trailing 14 days instead of all earlier days",
     **{**SPEC_DEFAULTS, "window_days": 14}},
    {"name": "E5_plus_scheduled_demand", "factor": "add scheduled origin-hour demand",
     **{**SPEC_DEFAULTS, "use_demand": True}},
    {"name": "E8_extended_history", "factor": "add 5 more past-only rate keys (airline x origin, origin x weekday, "
     "airline x block, dest x arrival hour, airline x route)", **{**SPEC_DEFAULTS, "history_set": "extended"}},
    {"name": "E9_recent7_history", "factor": "add short-memory (last 7 dates) rates for airline, origin, origin x block",
     **{**SPEC_DEFAULTS, "recent_days": RECENT_DAYS_CHOICE}},
    {"name": "E10_extra_demand", "factor": "add scheduled dest-hour, origin-day and previous-hour demand",
     **{**SPEC_DEFAULTS, "extra_demand": True}},
    {"name": "E11_schedule_extras", "factor": "add scheduled speed (distance / block time) and log distance",
     **{**SPEC_DEFAULTS, "sched_extras": True}},
]
SPEC_FACTOR_KEYS = ["use_history", "window_days", "warmup_days", "use_holiday", "use_demand",
                    "history_set", "recent_days", "extra_demand", "sched_extras"]

T0 = time.time()


class PipelineError(Exception):
    """A clear, expected failure (bad input, empty split, ...)."""


def log(message):
    print(message, flush=True)


def stamp(message):
    log(f"[{time.time() - T0:8.1f}s] {message}")


# --------------------------------------------------------------------------
# Arguments and run context
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Flight-delay MLlib pipeline (redesigned ML stage)")
    p.add_argument("--mode", choices=["select", "all"], default="all",
                   help="select: selection uses pre-test labels only (FlightDate dates define the cutoff). "
                        "all: select, freeze decisions, then read/score the test window once.")
    p.add_argument("--namespace", choices=["experiments", "production"], default="experiments",
                   help="experiments (default) never touches the baseline outputs.")
    p.add_argument("--confirm-overwrite-baseline", action="store_true",
                   help="required with --namespace production (overwrites the baseline outputs)")
    p.add_argument("--run-tag", default="", help="optional sub-folder name inside the experiments namespace")
    p.add_argument("--experiments", default="all",
                   help="comma-separated experiment names to run (E0_reference is always run)")
    p.add_argument("--extra-models", default="",
                   help="comma list of extra model families to tune and evaluate like LR/RF: gbt,mlp,xgb "
                        "(resource permitting; untested on 512 MB executors)")
    # Path overrides (used by the synthetic tests; any Hadoop-compatible URI works).
    p.add_argument("--clean-path", default=CLEAN_PATH)
    p.add_argument("--holidays-path", default=HOLIDAYS_PATH)
    p.add_argument("--hdfs-results", default="")
    p.add_argument("--hdfs-models", default="")
    p.add_argument("--local-results", default="")
    return p.parse_args(argv)


def _normalised_path_parts(path):
    """Return comparable components for local paths and Hadoop-style URIs."""
    from urllib.parse import urlsplit
    import posixpath

    value = str(path).replace("\\", "/").strip()
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        normal_path = posixpath.normpath(parsed.path or "/")
        return ("uri", parsed.scheme.lower(), parsed.netloc.lower(), normal_path)
    normal_local = posixpath.normpath(value)
    return ("local", normal_local)


def _is_same_or_descendant(path, protected_root):
    """True if a destination equals or is nested under a protected output root."""
    candidate = _normalised_path_parts(path)
    root = _normalised_path_parts(protected_root)
    if candidate[0] != root[0]:
        return False
    if candidate[0] == "uri":
        if candidate[1:3] != root[1:3]:
            return False
        candidate_path, root_path = candidate[3], root[3]
    else:
        candidate_path, root_path = candidate[1], root[1]
    root_path = root_path.rstrip("/") or "/"
    return candidate_path == root_path or candidate_path.startswith(root_path.rstrip("/") + "/")


class Ctx:
    def __init__(self, args):
        self.mode = args.mode
        self.namespace = args.namespace
        extras = [e.strip().lower() for e in args.extra_models.split(",") if e.strip()]
        unknown_extras = [e for e in extras if e not in EXTRA_MODEL_NAMES]
        if unknown_extras:
            raise PipelineError(f"Unknown --extra-models {unknown_extras}; valid: {sorted(EXTRA_MODEL_NAMES)}")
        self.extra_models = [e for e in EXTRA_MODEL_NAMES if e in extras]
        self.clean_path = args.clean_path
        self.holidays_path = args.holidays_path
        tag = args.run_tag.strip()
        if tag and not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise PipelineError("--run-tag may only contain letters, digits, '_' and '-'.")
        if args.namespace == "production":
            hdfs_results, hdfs_models, local = PRODUCTION_HDFS_RESULTS, PRODUCTION_HDFS_MODELS, PRODUCTION_LOCAL_RESULTS
        else:
            suffix = f"/{tag}" if tag else ""
            hdfs_results = f"{EXPERIMENT_HDFS_ROOT}{suffix}/results"
            hdfs_models = f"{EXPERIMENT_HDFS_ROOT}{suffix}/models"
            local = f"{EXPERIMENT_LOCAL_ROOT}{suffix}"
        self.hdfs_results = args.hdfs_results or hdfs_results
        self.hdfs_models = args.hdfs_models or hdfs_models
        self.local_results = args.local_results or local
        requested = [s.strip() for s in args.experiments.split(",") if s.strip()]
        all_names = [s["name"] for s in EXPERIMENT_SPECS]
        if requested == ["all"] or not requested:
            self.experiment_names = all_names
        else:
            unknown = [n for n in requested if n not in all_names]
            if unknown:
                raise PipelineError(f"Unknown experiment name(s) {unknown}; valid: {all_names}")
            self.experiment_names = [n for n in all_names if n in requested or n == "E0_reference"]
        path_overrides = bool(args.hdfs_results or args.hdfs_models or args.local_results)
        protected_targets = [
            ("HDFS results", PRODUCTION_HDFS_RESULTS, self.hdfs_results),
            ("HDFS models", PRODUCTION_HDFS_MODELS, self.hdfs_models),
            ("local results", PRODUCTION_LOCAL_RESULTS, self.local_results),
        ]
        touched_production = [
            label for label, protected_root, resolved_path in protected_targets
            if _is_same_or_descendant(resolved_path, protected_root)
        ]

        # Production mode is deliberately narrow: only the exact production destinations,
        # all-mode (which scores the held-out test after decisions are frozen), and an
        # explicit confirmation are permitted. Path overrides must use the experiments
        # namespace instead; otherwise one overridden path could bypass the old guard.
        if args.namespace == "production":
            if args.mode != "all":
                raise PipelineError("--namespace production requires --mode all.")
            if not args.confirm_overwrite_baseline:
                raise PipelineError(
                    "Refusing to write to production paths without --confirm-overwrite-baseline. "
                    "Back up local results and HDFS models first.")
            if path_overrides:
                raise PipelineError(
                    "Path overrides are not allowed with --namespace production. "
                    "Use --namespace experiments and a unique --run-tag for trials.")
        elif touched_production:
            raise PipelineError(
                "Refusing to write to protected production destination(s) "
                f"{touched_production} from namespace={args.namespace!r}. "
                "Use the experiments namespace and a unique --run-tag.")


def assert_no_leakage(source_columns=None):
    """Hard check that no predictor source column is an outcome / post-departure field."""
    columns = FEATURE_SOURCE_COLUMNS if source_columns is None else source_columns
    bad = sorted(c for c in columns if c in FORBIDDEN_SOURCE_COLUMNS or c.startswith("Div"))
    if bad:
        raise PipelineError(f"Leakage guard: forbidden source columns in feature list: {bad}")


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def preflight_local_output(ctx):
    os.makedirs(ctx.local_results, exist_ok=True)
    probe = os.path.join(ctx.local_results, ".write_probe")
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write("ok")
    os.remove(probe)


def save_table(spark, ctx, filename, header, rows):
    """Write a small result table to the local results folder and to HDFS (one CSV part file)."""
    local_path = os.path.join(ctx.local_results, filename)
    with open(local_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    string_rows = [tuple("" if v is None else str(v) for v in row) for row in rows]
    schema = T.StructType([T.StructField(name, T.StringType(), True) for name in header])
    hdfs_path = f"{ctx.hdfs_results}/{filename[:-4] if filename.endswith('.csv') else filename}"
    (spark.createDataFrame(string_rows, schema)
     .coalesce(1).write.mode("overwrite").option("header", "true").csv(hdfs_path))
    log(f"Saved {filename}: local={local_path} hdfs={hdfs_path}")


def save_text(spark, ctx, relative_name, text):
    path = f"{ctx.hdfs_results}/{relative_name}"
    spark.createDataFrame([(text,)], ["value"]).coalesce(1).write.mode("overwrite").text(path)
    return path


def write_local_text(ctx, filename, text):
    with open(os.path.join(ctx.local_results, filename), "w", encoding="utf-8") as handle:
        handle.write(text)


# --------------------------------------------------------------------------
# Input validation, eligibility and split helpers (pure / cheap, unit-tested)
# --------------------------------------------------------------------------
def validate_schema(schema):
    """Return a list of human-readable problems with the clean Parquet schema."""
    problems = []
    types = {f.name: f.dataType.simpleString() for f in schema.fields}
    for column in REQUIRED_CLEAN_COLUMNS:
        if column not in types:
            problems.append(f"missing column {column}")
            continue
        if types[column] not in REQUIRED_COLUMN_KINDS[column]:
            problems.append(f"column {column} has type {types[column]}, expected one of "
                            f"{sorted(REQUIRED_COLUMN_KINDS[column])}")
    return problems


def eligibility_category_expr():
    """Mutually exclusive eligibility category per row (first matching rule wins)."""
    return (
        F.when(F.col("Cancelled").isNull() | F.col("Diverted").isNull(), "missing_cancelled_or_diverted_flag")
        .when(F.col("Cancelled") == 1, "cancelled")
        .when(F.col("Diverted") == 1, "diverted")
        .when(F.col("ArrDel15").isNull(), "missing_target")
        .when(~F.col("ArrDel15").isin(0, 1), "invalid_target")
        .when((F.col("Cancelled") == 0) & (F.col("Diverted") == 0), "eligible")
        .otherwise("other_flag_value")
    )


ELIGIBILITY_ORDER = ["cancelled", "diverted", "missing_cancelled_or_diverted_flag", "missing_target",
                     "invalid_target", "other_flag_value", "eligible"]


def eligible_condition():
    return (F.col("Cancelled") == 0) & (F.col("Diverted") == 0) & F.col("ArrDel15").isin(0, 1)


def compute_windows(dates):
    """Chronological windows by whole dates. `dates` must be a sorted list of distinct dates."""
    if len(dates) < 3:
        raise PipelineError("Fewer than three distinct flight dates; cannot make train/validation/test splits.")
    val_index = max(1, min(int(len(dates) * VALIDATION_START_DATE_FRACTION), len(dates) - 2))
    test_index = max(val_index + 1, min(int(len(dates) * TEST_START_DATE_FRACTION), len(dates) - 1))
    return {
        "train": (dates[0], dates[val_index - 1]),
        "validation": (dates[val_index], dates[test_index - 1]),
        "test": (dates[test_index], dates[-1]),
    }


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------
def minute_of_day(column):
    """Parse an HHMM integer (e.g. 1745, 2400) into minutes after midnight, else null."""
    hh = F.floor(column / F.lit(100))
    mm = column - hh * F.lit(100)
    valid = column.isNotNull() & (column >= 0) & (column <= 2400) & (mm >= 0) & (mm < 60)
    return F.when(valid, F.pmod(hh * F.lit(60) + mm, F.lit(1440))).cast("double")


def compute_demand_table(flights, cutoff_date=None):
    """Scheduled flights per (date, origin, scheduled departure hour).

    Counts ALL scheduled flights, including cancelled and diverted ones: whether a flight
    was later cancelled is an outcome, so excluding those rows would leak. Uses only
    schedule fields; no delay information. One row per key (join cannot duplicate rows).
    """
    base = flights.select("FlightDate", "Origin", "CRSDepTime")
    if cutoff_date is not None:
        base = base.filter(F.col("FlightDate") < F.lit(cutoff_date))
    hour = F.coalesce(F.floor(minute_of_day(F.col("CRSDepTime")) / F.lit(60.0)).cast("int"), F.lit(-1))
    return (base.withColumn("_dep_hour_key", hour)
            .groupBy("FlightDate", "Origin", "_dep_hour_key")
            .agg(F.count(F.lit(1)).alias("_sched_n")))


def compute_dest_demand_table(flights, cutoff_date=None):
    """Scheduled arrivals per (departure date, destination, scheduled arrival hour).

    All scheduled flights are counted, including cancelled and diverted ones (see
    compute_demand_table). Keyed by the DEPARTURE date; for overnight flights the arrival hour
    falls on the next calendar day, which this approximation ignores.
    """
    base = flights.select("FlightDate", "Dest", "CRSArrTime")
    if cutoff_date is not None:
        base = base.filter(F.col("FlightDate") < F.lit(cutoff_date))
    hour = F.coalesce(F.floor(minute_of_day(F.col("CRSArrTime")) / F.lit(60.0)).cast("int"), F.lit(-1))
    return (base.withColumn("_arr_hour_key", hour)
            .groupBy("FlightDate", "Dest", "_arr_hour_key")
            .agg(F.count(F.lit(1)).alias("_dest_sched_n")))


def build_eligible_frame(flights, holidays, demand, cutoff_date=None):
    """Eligible flights with all pre-departure features (spec-independent)."""
    f = flights.filter(eligible_condition())
    if cutoff_date is not None:
        f = f.filter(F.col("FlightDate") < F.lit(cutoff_date))
    two_pi_per_day = 2.0 * math.pi / 1440.0
    dest_demand = compute_dest_demand_table(flights, cutoff_date)
    origin_day = demand.groupBy("FlightDate", "Origin").agg(F.sum("_sched_n").alias("_origin_day_n"))
    prev_hour = (demand.filter(F.col("_dep_hour_key") >= 0)
                 .select("FlightDate", "Origin", (F.col("_dep_hour_key") + 1).alias("_dep_hour_key"),
                         F.col("_sched_n").alias("_prev_hour_n")))
    out = (
        f.withColumn("_dep_min", minute_of_day(F.col("CRSDepTime")))
        .withColumn("_arr_min", minute_of_day(F.col("CRSArrTime")))
        .withColumn("_dep_hour_key",
                    F.coalesce(F.floor(F.col("_dep_min") / F.lit(60.0)).cast("int"), F.lit(-1)))
        .withColumn("_arr_hour_key",
                    F.coalesce(F.floor(F.col("_arr_min") / F.lit(60.0)).cast("int"), F.lit(-1)))
        .join(F.broadcast(holidays), on="FlightDate", how="left")
        .join(demand, on=["FlightDate", "Origin", "_dep_hour_key"], how="left")
        .join(prev_hour, on=["FlightDate", "Origin", "_dep_hour_key"], how="left")
        .join(origin_day, on=["FlightDate", "Origin"], how="left")
        .join(dest_demand, on=["FlightDate", "Dest", "_arr_hour_key"], how="left")
        .withColumn("HolidayPeriod", F.coalesce(F.col("HolidayPeriod"), F.lit("Regular day")))
        .withColumn("label", F.col("ArrDel15").cast("double"))
        .withColumn("Reporting_Airline", F.coalesce(F.col("Reporting_Airline"), F.lit("UNKNOWN")))
        .withColumn("DepTimeBlk", F.coalesce(F.col("DepTimeBlk"), F.lit("UNKNOWN")))
        .withColumn("DayOfWeekCat",
                    F.coalesce(F.concat(F.lit("DOW_"), F.col("DayOfWeek").cast("string")), F.lit("DOW_UNKNOWN")))
        .withColumn("TimeParseInvalid", (F.col("_dep_min").isNull() | F.col("_arr_min").isNull()).cast("int"))
        .withColumn("DepMinuteSin", F.coalesce(F.sin(F.lit(two_pi_per_day) * F.col("_dep_min")), F.lit(0.0)))
        .withColumn("DepMinuteCos", F.coalesce(F.cos(F.lit(two_pi_per_day) * F.col("_dep_min")), F.lit(0.0)))
        .withColumn("ArrMinuteSin", F.coalesce(F.sin(F.lit(two_pi_per_day) * F.col("_arr_min")), F.lit(0.0)))
        .withColumn("ArrMinuteCos", F.coalesce(F.cos(F.lit(two_pi_per_day) * F.col("_arr_min")), F.lit(0.0)))
        .withColumn("ScheduledDepHour", F.coalesce(F.floor(F.col("_dep_min") / F.lit(60.0)).cast("double"), F.lit(0.0)))
        .withColumn("ScheduledArrHour", F.coalesce(F.floor(F.col("_arr_min") / F.lit(60.0)).cast("double"), F.lit(0.0)))
        .withColumn("IsHolidayWindow", (F.col("HolidayPeriod") != F.lit("Regular day")).cast("double"))
        .withColumn("OriginHourDemandLog", F.log1p(F.coalesce(F.col("_sched_n"), F.lit(0)).cast("double")))
        .withColumn("DestHourDemandLog", F.log1p(F.coalesce(F.col("_dest_sched_n"), F.lit(0)).cast("double")))
        .withColumn("OriginDayDemandLog", F.log1p(F.coalesce(F.col("_origin_day_n"), F.lit(0)).cast("double")))
        .withColumn("OriginPrevHourDemandLog", F.log1p(F.coalesce(F.col("_prev_hour_n"), F.lit(0)).cast("double")))
        .withColumn("SchedSpeedMph", F.when(F.col("CRSElapsedTime") > 0,
                                            F.col("Distance") / (F.col("CRSElapsedTime") / F.lit(60.0))))
        .withColumn("DistanceLog", F.when(F.col("Distance") >= 0, F.log1p(F.col("Distance"))))
        .select(
            "FlightDate", "label", "HolidayPeriod", "Reporting_Airline", "Origin", "Dest",
            "DepTimeBlk", "DayOfWeekCat", "CRSElapsedTime", "Distance",
            *BASE_NUMERIC_FEATURES, *HOLIDAY_NUMERIC_FEATURES, *DEMAND_FEATURES, *EXTRA_DEMAND_FEATURES,
            *SCHED_EXTRA_FEATURES, "TimeParseInvalid",
        )
    )
    return out


# --------------------------------------------------------------------------
# Historical-rate features (the leakage-sensitive part)
# --------------------------------------------------------------------------
def _day_number():
    return F.datediff(F.col("FlightDate"), F.lit("1970-01-01").cast("date"))


def _past_window(partition_cols, window_days):
    """Window over STRICTLY EARLIER dates of per-date aggregates (one row per key and date)."""
    base = Window.partitionBy(*partition_cols) if partition_cols else Window
    if window_days is None:
        return base.orderBy("FlightDate").rowsBetween(Window.unboundedPreceding, -1)
    return base.orderBy("_day").rangeBetween(-int(window_days), -1)


def build_history_tables(fit_base, keep, window_days=None, keys_map=None):
    """Per-key (positives, flights) counts and the prior, from the fitting window only.

    With window_days, only the last `window_days` dates of the fitting window are used
    (the same trailing window a training row on the next date would see).
    """
    keys_map = HISTORY_KEYS if keys_map is None else keys_map
    base = fit_base
    if window_days is not None:
        max_date = fit_base.agg(F.max("FlightDate")).first()[0]
        base = fit_base.filter(F.col("FlightDate") > F.date_sub(F.lit(max_date), int(window_days)))
    prior = float(base.agg(F.avg("label")).first()[0])
    tables = {}
    for name, keys in keys_map.items():
        table = keep(
            base.groupBy(*keys).agg(F.sum("label").alias("hist_pos"), F.count(F.lit(1)).alias("hist_n"))
        )
        log(f"  historical-rate table {name} keys={keys}: {table.count():,} groups")
        tables[name] = table
    return prior, tables


def apply_history(df, prior, tables, keys_map=None):
    """Apply smoothed rates from a completed, strictly earlier fitting window.

    For validation/test scoring rows ONLY. The tables come from a window that ended before
    these rows occur, so all labels they contain were historically available. The scoring
    rows' own labels are never read. Keys are unique in each table, so the left join
    cannot duplicate rows (callers verify the row count).
    """
    keys_map = HISTORY_KEYS if keys_map is None else keys_map
    out = df
    pseudo_positives = F.lit(float(HISTORY_SMOOTHING * prior))
    pseudo_flights = F.lit(float(HISTORY_SMOOTHING))
    for name, keys in keys_map.items():
        table = (tables[name]
                 .withColumnRenamed("hist_pos", f"{name}_pos")
                 .withColumnRenamed("hist_n", f"{name}_n"))
        out = out.join(table, on=keys, how="left")
        positives = F.coalesce(F.col(f"{name}_pos"), F.lit(0.0))
        flights = F.coalesce(F.col(f"{name}_n"), F.lit(0)).cast("double")
        out = out.withColumn(name, (positives + pseudo_positives) / (flights + pseudo_flights))
        out = out.drop(f"{name}_pos", f"{name}_n")
    return out


def apply_history_training_past_only(fit_base, window_days=None, keys_map=None):
    """Create fitting-row history features using strictly earlier flight dates.

    For every fitting row on date D, each category rate uses observations from dates
    strictly before D (all earlier dates, or the trailing `window_days` dates before D).
    The global prior likewise uses only earlier dates; the first date has no earlier data
    and uses a neutral 0.5. A leave-one-day-out aggregate over the whole fitting window
    would let an October row see November labels; this cannot.
    """
    global_daily = (
        fit_base.groupBy("FlightDate")
        .agg(F.sum("label").alias("_global_day_pos"), F.count(F.lit(1)).alias("_global_day_n"))
        .withColumn("_day", _day_number())
    )
    gw = _past_window([], window_days)
    global_daily = (
        global_daily
        .withColumn("_global_prev_pos", F.sum("_global_day_pos").over(gw))
        .withColumn("_global_prev_n", F.sum("_global_day_n").over(gw))
        .withColumn(
            "_global_prior",
            F.when(F.coalesce(F.col("_global_prev_n"), F.lit(0)) > 0,
                   F.col("_global_prev_pos") / F.col("_global_prev_n")).otherwise(F.lit(0.5)),
        )
        .select("FlightDate", "_global_prior")
    )

    keys_map = HISTORY_KEYS if keys_map is None else keys_map
    out = fit_base
    for name, keys in keys_map.items():
        kw = _past_window(keys, window_days)
        daily_rates = (
            fit_base.groupBy(*(keys + ["FlightDate"]))
            .agg(F.sum("label").alias("_day_pos"), F.count(F.lit(1)).alias("_day_n"))
            .withColumn("_day", _day_number())
            .withColumn("_prev_pos", F.sum("_day_pos").over(kw))
            .withColumn("_prev_n", F.sum("_day_n").over(kw))
            .join(F.broadcast(global_daily), on="FlightDate", how="left")
        )
        prev_pos = F.coalesce(F.col("_prev_pos"), F.lit(0.0))
        prev_n = F.coalesce(F.col("_prev_n"), F.lit(0)).cast("double")
        prior = F.coalesce(F.col("_global_prior"), F.lit(0.5))
        rate = (prev_pos + F.lit(HISTORY_SMOOTHING) * prior) / (prev_n + F.lit(HISTORY_SMOOTHING))
        daily_rates = daily_rates.select(*(keys + ["FlightDate"]), rate.alias(name))
        out = out.join(daily_rates, on=keys + ["FlightDate"], how="left")
    return out


# --------------------------------------------------------------------------
# Feature specs, preprocessing pipeline
# --------------------------------------------------------------------------
def spec_columns(spec):
    """(impute, plain numeric, history, categorical) column lists for a feature spec."""
    plain = list(BASE_NUMERIC_FEATURES)
    categorical = list(BASE_CATEGORICAL_FEATURES)
    if spec["use_holiday"]:
        plain += HOLIDAY_NUMERIC_FEATURES
        categorical += HOLIDAY_CATEGORICAL_FEATURES
    if spec["use_demand"]:
        plain += DEMAND_FEATURES
    if spec["extra_demand"]:
        plain += EXTRA_DEMAND_FEATURES
    impute = list(IMPUTE_FEATURES) + (SCHED_EXTRA_FEATURES if spec["sched_extras"] else [])
    history = [name for group in history_groups(spec) for name in group["keys_map"]]
    return impute, plain, history, categorical


def history_groups(spec):
    """History feature groups for a spec: [{name, keys_map, window_days}, ...] (empty if no history)."""
    if not spec["use_history"]:
        return []
    base_keys = dict(HISTORY_KEYS)
    if spec["history_set"] == "extended":
        base_keys.update(EXTRA_HISTORY_KEYS)
    groups = [{"name": "base", "keys_map": base_keys, "window_days": spec["window_days"]}]
    if spec["recent_days"]:
        groups.append({"name": "recent", "keys_map": dict(RECENT_HISTORY_KEYS),
                       "window_days": int(spec["recent_days"])})
    return groups


def spec_label(spec):
    if not spec["use_history"]:
        history = "off"
    elif spec["window_days"]:
        history = "trailing" + str(spec["window_days"]) + "d"
    else:
        history = "expanding"
    holiday = "on" if spec["use_holiday"] else "off"
    demand = "on" if spec["use_demand"] else "off"
    extras = (f",historyset={spec['history_set']},recent={spec['recent_days'] or 'off'},"
              f"extrademand={'on' if spec['extra_demand'] else 'off'},"
              f"schedextras={'on' if spec['sched_extras'] else 'off'}")
    return f"history={history},warmup={spec['warmup_days']},holiday={holiday},demand={demand}{extras}"


def make_pipeline(spec):
    """Preprocessing fitted on the fitting window only (imputer, indexers, one-hot, assembler)."""
    impute, plain, history, categorical = spec_columns(spec)
    imputed = [f"{c}__imputed" for c in impute]
    index_cols = [f"{c}__idx" for c in categorical]
    ohe_cols = [f"{c}__ohe" for c in categorical]
    stages = [Imputer(strategy="median", inputCols=impute, outputCols=imputed)]
    stages.extend(StringIndexer(inputCol=c, outputCol=i, handleInvalid="keep")
                  for c, i in zip(categorical, index_cols))
    stages.append(OneHotEncoder(inputCols=index_cols, outputCols=ohe_cols, handleInvalid="keep", dropLast=False))
    stages.append(VectorAssembler(inputCols=imputed + plain + history + ohe_cols,
                                  outputCol="features", handleInvalid="keep"))
    return Pipeline(stages=stages)


def feature_names(vec_df):
    """Recover VectorAssembler feature names from column metadata."""
    metadata = vec_df.schema["features"].metadata.get("ml_attr", {})
    names = [f"feature_{i}" for i in range(int(metadata.get("num_attrs", 0)))]
    for attr_list in metadata.get("attrs", {}).values():
        for item in attr_list:
            idx = int(item.get("idx", -1))
            if 0 <= idx < len(names):
                names[idx] = item.get("name", names[idx])
    return names


def check_rows(label, expected, df_or_count):
    actual = df_or_count if isinstance(df_or_count, int) else df_or_count.count()
    if actual != expected:
        raise PipelineError(f"{label}: row count changed from {expected:,} to {actual:,} "
                            f"(a join duplicated or dropped flights).")


def prepare_stage(label, spec, fit_base, apply_bases, keep, release):
    """Build feature vectors for one fitting window under one feature spec."""
    stamp(f"{label}: preparing features [{spec_label(spec)}]")
    fit_n = fit_base.count()
    prior, tables, priors = None, {}, {}
    groups = history_groups(spec)
    fit_all = fit_base
    for group in groups:
        group_prior, group_tables = build_history_tables(
            fit_base, keep, group["window_days"], group["keys_map"])
        priors[group["name"]] = group_prior
        group["prior"], group["tables"] = group_prior, group_tables
        tables.update(group_tables)
        fit_all = apply_history_training_past_only(fit_all, group["window_days"], group["keys_map"])
    if groups:
        fit_all = keep(fit_all)
        check_rows(f"{label} history join (fitting rows)", fit_n, fit_all)
        prior = priors["base"]
    else:
        prior = float(fit_base.agg(F.avg("label")).first()[0])
    fit_h = fit_all
    applied_h = {}
    for name, df in apply_bases.items():
        n_before = df.count()
        out = df
        for group in groups:
            out = apply_history(out, group["prior"], group["tables"], group["keys_map"])
        applied_h[name] = keep(out) if groups else out
        check_rows(f"{label} history join ({name} rows)", n_before, applied_h[name])

    warmup = int(spec["warmup_days"])
    if warmup > 0:
        first_date = fit_base.agg(F.min("FlightDate")).first()[0]
        fit_h = keep(fit_h.filter(F.col("FlightDate") >= F.date_add(F.lit(first_date), warmup)))
    fit_rows = fit_h.count()
    if fit_rows == 0:
        raise PipelineError(f"{label}: no fitting rows left after warm-up exclusion.")

    pipeline_model = make_pipeline(spec).fit(fit_h)

    def vectorize(df):
        return pipeline_model.transform(df).select("label", "HolidayPeriod", "features")

    fit_vec = keep(vectorize(fit_h))
    fit_vec.count()
    applied_vec = {}
    for name, df in applied_h.items():
        applied_vec[name] = keep(vectorize(df))
        applied_vec[name].count()
    for df in {id(d): d for d in (fit_all, fit_h)}.values():
        if df is not fit_base:
            release(df)
    for name, df in applied_h.items():
        if df is not apply_bases[name]:
            release(df)
    names = feature_names(fit_vec)
    stamp(f"{label}: feature vectors ready ({fit_rows:,} fitting rows, {len(names)} features)")
    return {"spec": spec, "prior": prior, "priors": priors, "tables": tables, "pipeline_model": pipeline_model,
            "fit_vec": fit_vec, "applied": applied_vec, "feature_names": names, "fit_rows": fit_rows}


def release_stage(stage, release):
    if stage is None:
        return
    release(stage["fit_vec"])
    for df in stage["applied"].values():
        release(df)
    for table in stage["tables"].values():
        release(table)


# --------------------------------------------------------------------------
# Models, scoring and metrics
# --------------------------------------------------------------------------
def fit_prevalence(vec_df):
    return float(vec_df.agg(F.avg("label")).first()[0])


def balanced_pos_weight(prevalence):
    return (1.0 - prevalence) / prevalence


def fit_logistic(vec_df, reg_param, pos_weight=None):
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=LR_MAX_ITER,
                            regParam=float(reg_param), elasticNetParam=0.0, standardization=True)
    if pos_weight is not None:
        vec_df = vec_df.withColumn(
            "weight", F.when(F.col("label") == 1.0, F.lit(float(pos_weight))).otherwise(F.lit(1.0)))
        lr = lr.setWeightCol("weight")
    return lr.fit(vec_df)


def fit_forest(vec_df, config):
    return RandomForestClassifier(
        labelCol="label", featuresCol="features", numTrees=RF_NUM_TREES,
        maxDepth=int(config["maxDepth"]), minInstancesPerNode=int(config["minInstancesPerNode"]),
        maxBins=RF_MAX_BINS, subsamplingRate=RF_SUBSAMPLING_RATE,
        featureSubsetStrategy=RF_FEATURE_SUBSET, maxMemoryInMB=RF_MAX_MEMORY_MB, seed=SEED,
    ).fit(vec_df)


def fit_gbt(vec_df, config):
    return GBTClassifier(labelCol="label", featuresCol="features", maxIter=int(config["maxIter"]),
                         maxDepth=int(config["maxDepth"]), stepSize=GBT_STEP_SIZE,
                         minInstancesPerNode=GBT_MIN_INSTANCES, maxBins=RF_MAX_BINS, subsamplingRate=0.5,
                         maxMemoryInMB=RF_MAX_MEMORY_MB, seed=SEED).fit(vec_df)


def fit_mlp(vec_df, config):
    """Small feed-forward neural network (MLlib MultilayerPerceptronClassifier).

    Neural nets need scaled inputs, so a StandardScaler (fitted on the fitting rows only) is
    part of the returned PipelineModel; it is applied automatically when scoring.
    """
    n_features = len(feature_names(vec_df))
    layers = [n_features] + [int(h) for h in config["hidden"]] + [2]
    pipeline = Pipeline(stages=[
        StandardScaler(inputCol="features", outputCol="features_scaled", withMean=False, withStd=True),
        MultilayerPerceptronClassifier(labelCol="label", featuresCol="features_scaled", layers=layers,
                                       maxIter=MLP_MAX_ITER, blockSize=MLP_BLOCK_SIZE, seed=SEED),
    ])
    return pipeline.fit(vec_df)


def fit_xgb(vec_df, config):
    """Distributed CPU XGBoost; optional because it adds Python dependencies to every YARN node."""
    if SparkXGBClassifier is None:
        raise PipelineError(
            "--extra-models xgb requires xgboost-cpu, scikit-learn, scipy and pyarrow "
            "on the driver and every executor"
        )
    return SparkXGBClassifier(
        label_col="label", features_col="features", probability_col="probability",
        prediction_col="prediction", raw_prediction_col="rawPrediction",
        num_workers=XGB_NUM_WORKERS, n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]), learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=0.0,
        eval_metric="aucpr", tree_method="hist",
        max_bin=128, random_state=SEED, force_repartition=True, verbosity=1,
    ).fit(vec_df)


def tune_candidates(family, configs, label_fn, fit_fn, train_vec, val_vec, keep, release,
                    search_rows, experiment_rows, spec, description):
    """Validation-only grid for one model family; returns the best candidate dict."""
    best = None
    for config in configs:
        started = time.time()
        label = label_fn(config)
        stamp(f"{family} candidate {label}")
        scored = score_dataframe(fit_fn(train_vec, config), val_vec, keep)
        roc, pr = auc_pair(scored)
        search_rows.append([family, label, roc, pr])
        experiment_rows.append({"Experiment": f"tune_{family}", "Description": description, "Spec": spec,
                                "Model": f"{family}({label})", "Role": "model tuning",
                                "ROC": roc, "PR": pr, "Seconds": time.time() - started})
        log(f"  validation AUC-ROC={roc:.5f} AUC-PR={pr:.5f}")
        if best is None or pr > best["pr"] + 1e-12:
            if best is not None:
                release(best["scored"])
            best = {"config": config, "pr": pr, "roc": roc, "scored": scored, "label": label}
        else:
            release(scored)
    log(f"Selected {family}: {best['label']} (validation AUC-PR={best['pr']:.5f})")
    return best


def score_dataframe(model, vec_df, keep):
    """Class-1 probability per flight; cached because several metrics reuse it."""
    scored = keep(
        model.transform(vec_df).select(
            "label", "HolidayPeriod", "probability",
            vector_to_array(F.col("probability"))[1].alias("score"),
        )
    )
    scored.count()
    return scored


def auc_pair(scored):
    roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC").evaluate(scored)
    pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR").evaluate(scored)
    return float(roc), float(pr)


def prf(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return precision, recall, f1, accuracy


def confusion(scored, threshold):
    predicted = F.col("score") >= F.lit(float(threshold))
    row = scored.agg(
        F.sum(F.when(predicted & (F.col("label") == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when(predicted & (F.col("label") == 0.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((~predicted) & (F.col("label") == 1.0), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((~predicted) & (F.col("label") == 0.0), 1).otherwise(0)).alias("tn"),
    ).first()
    return tuple(int(row[k] or 0) for k in ("tp", "fp", "fn", "tn"))


def threshold_search(scored_validation, candidates):
    """Validation-only threshold grid. Returns (best_threshold, rows)."""
    rows = []
    for start in range(0, len(candidates), 12):
        chunk = candidates[start:start + 12]
        expressions = []
        for i, threshold in enumerate(chunk):
            predicted = F.col("score") >= F.lit(float(threshold))
            expressions.append(F.sum(F.when(predicted & (F.col("label") == 1.0), 1).otherwise(0)).alias(f"tp_{i}"))
            expressions.append(F.sum(F.when(predicted & (F.col("label") == 0.0), 1).otherwise(0)).alias(f"fp_{i}"))
            expressions.append(F.sum(F.when((~predicted) & (F.col("label") == 1.0), 1).otherwise(0)).alias(f"fn_{i}"))
        totals = scored_validation.agg(*expressions).first()
        for i, threshold in enumerate(chunk):
            tp, fp, fn = (int(totals[f"{k}_{i}"] or 0) for k in ("tp", "fp", "fn"))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append([float(threshold), tp, fp, fn, precision, recall, f1])
    best = max(rows, key=lambda r: (r[6], r[4], r[0]))   # F1, then precision, then higher threshold
    return best[0], rows


def ranking_metrics(scored, fractions, total, positives):
    """Precision/recall/lift when flagging the riskiest X% of flights (labels not used to rank)."""
    cutoffs = scored.approxQuantile("score", [1.0 - f for f in fractions], 0.001)
    expressions = []
    for i, cutoff in enumerate(cutoffs):
        flagged = F.col("score") >= F.lit(float(cutoff))
        expressions.append(F.sum(F.when(flagged, 1).otherwise(0)).alias(f"n_{i}"))
        expressions.append(F.sum(F.when(flagged & (F.col("label") == 1.0), 1).otherwise(0)).alias(f"p_{i}"))
    totals = scored.agg(*expressions).first()
    prevalence = positives / total if total else 0.0
    rows = []
    for i, fraction in enumerate(fractions):
        flagged_n = int(totals[f"n_{i}"] or 0)
        flagged_p = int(totals[f"p_{i}"] or 0)
        precision = flagged_p / flagged_n if flagged_n else 0.0
        recall = flagged_p / positives if positives else 0.0
        lift = precision / prevalence if prevalence else 0.0
        rows.append([fraction, flagged_n / total if total else 0.0, flagged_n, precision, recall, lift])
    return rows


# --------------------------------------------------------------------------
# Experiment helpers
# --------------------------------------------------------------------------
def combine_specs(reference, winners):
    """Merge the single-factor changes of every winning challenger into one spec."""
    merged = {k: reference[k] for k in SPEC_FACTOR_KEYS}
    for winner in winners:
        for key in SPEC_FACTOR_KEYS:
            if winner[key] != reference[key]:
                merged[key] = winner[key]
    if not merged["use_history"]:           # history-only factors are meaningless without history
        merged["window_days"], merged["warmup_days"] = None, 0
        merged["history_set"], merged["recent_days"] = "base", None
    merged["name"] = "E6_combined_winners"
    merged["factor"] = "combination of every challenger that beat the reference by the margin"
    return merged


def evaluate_spec(spec, train_base, validation_base, keep, release):
    """Fit LR (fixed regParam) on train under `spec`; score validation. Returns (result, stage)."""
    started = time.time()
    stage = prepare_stage(spec["name"], spec, train_base, {"validation": validation_base}, keep, release)
    model = fit_logistic(stage["fit_vec"], EXPERIMENT_LR_REG)
    scored = score_dataframe(model, stage["applied"]["validation"], keep)
    roc, pr = auc_pair(scored)
    threshold, rows = threshold_search(scored, THRESHOLD_CANDIDATES)
    best_f1 = max(r[6] for r in rows)
    release(scored)
    result = {
        "name": spec["name"], "spec": spec, "roc": roc, "pr": pr, "best_f1": best_f1,
        "best_threshold": threshold, "seconds": time.time() - started,
        "n_features": len(stage["feature_names"]),
    }
    log(f"  {spec['name']}: validation AUC-ROC={roc:.5f} AUC-PR={pr:.5f} bestF1={best_f1:.5f}@{threshold:.2f} "
        f"({result['seconds']:.0f}s)")
    return result, stage


def select_feature_spec(spec_names, train_base, validation_base, keep, release):
    """One-factor-at-a-time comparison against the reference. Validation data only."""
    specs = [s for s in EXPERIMENT_SPECS if s["name"] in spec_names]
    reference = specs[0]
    results = {}
    stages = {}

    def prune(keep_names):
        for name in list(stages):
            if name not in keep_names:
                release_stage(stages.pop(name), release)

    results[reference["name"]], stages[reference["name"]] = evaluate_spec(
        reference, train_base, validation_base, keep, release)
    ref_pr = results[reference["name"]]["pr"]

    def beats(res):
        return res["pr"] >= ref_pr + SELECTION_MARGIN

    def best_challenger():
        eligible = [r for n, r in results.items() if n != reference["name"] and beats(r)]
        return max(eligible, key=lambda r: r["pr"]) if eligible else None

    for spec in specs[1:]:
        results[spec["name"]], stages[spec["name"]] = evaluate_spec(
            spec, train_base, validation_base, keep, release)
        best = best_challenger()
        prune({reference["name"]} | ({best["name"]} if best else set()))

    winners = [r["spec"] for n, r in results.items() if n != reference["name"] and beats(r)]
    if len(winners) >= 2:
        combined = combine_specs(reference, winners)
        results[combined["name"]], stages[combined["name"]] = evaluate_spec(
            combined, train_base, validation_base, keep, release)
        best = best_challenger()
        prune({reference["name"]} | ({best["name"]} if best else set()))

    best = best_challenger()
    chosen_name = best["name"] if best else reference["name"]
    prune({chosen_name})
    log(f"Feature selection (validation AUC-PR, margin {SELECTION_MARGIN}): reference={ref_pr:.5f}; "
        f"chosen={chosen_name}")
    return results[chosen_name]["spec"], stages[chosen_name], results


def tidy_spec_fields(spec):
    return [
        "yes" if spec["use_history"] else "no",
        spec["window_days"] if spec["window_days"] else ("all earlier dates" if spec["use_history"] else ""),
        spec["warmup_days"],
        "yes" if spec["use_holiday"] else "no",
        "yes" if spec["use_demand"] else "no",
        spec["history_set"],
        spec["recent_days"] if spec["recent_days"] else "",
        "yes" if spec["extra_demand"] else "no",
        "yes" if spec["sched_extras"] else "no",
    ]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None, spark=None) -> int:
    owns_spark = spark is None
    cached = []

    def keep(df):
        df.persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(df)
        return df

    def release(df):
        try:
            df.unpersist()
        except Exception:
            pass
        cached[:] = [d for d in cached if d is not df]

    try:
        args = parse_args(argv)
        ctx = Ctx(args)
        assert_no_leakage()
        preflight_local_output(ctx)
        if spark is None:
            builder = SparkSession.builder.appName("BDA-03-Flight-Delay-MLlib")
            for key, value in SPARK_SESSION_CONF.items():
                builder = builder.config(key, value)
            spark = builder.getOrCreate()
            spark.sparkContext.setLogLevel("WARN")
        log(f"Spark version: {spark.version}")
        log(f"ML_MODE={ctx.mode} ML_NAMESPACE={ctx.namespace}")
        log(f"Results: local={ctx.local_results} hdfs={ctx.hdfs_results} models={ctx.hdfs_models}")
        log(f"Predictor source columns (allow-list): {FEATURE_SOURCE_COLUMNS}")
        if ctx.mode == "select":
            log("MODE select: split cutoffs use FlightDate only; all label-based accounting, fitting, "
                "tuning and threshold selection are restricted to dates before the test cutoff.")

        # ---------------- inputs ----------------
        stamp(f"Reading cleaned Parquet: {ctx.clean_path}")
        flights = spark.read.parquet(ctx.clean_path)
        problems = validate_schema(flights.schema)
        if problems:
            raise PipelineError("Cleaned Parquet schema problems: " + "; ".join(problems))

        stamp(f"Reading holiday calendar: {ctx.holidays_path}")
        holidays = (
            spark.read.option("header", "true").csv(ctx.holidays_path)
            .select("FlightDate", "HolidayPeriod")
            .withColumn("FlightDate", F.to_date("FlightDate", "yyyy-MM-dd"))
            .withColumn("HolidayPeriod", F.trim(F.col("HolidayPeriod")))
            .filter(F.col("FlightDate").isNotNull() & F.col("HolidayPeriod").isNotNull())
        )
        if holidays.groupBy("FlightDate").count().filter(F.col("count") > 1).limit(1).count():
            raise PipelineError("holidays.csv contains duplicate FlightDate values; join would duplicate flights.")

        # ---------------- split boundaries from dates only ----------------
        # FlightDate is used only to define the chronological cutoffs. Do NOT use
        # eligible_condition() here: it references outcome/eligibility columns.
        dates = [r[0] for r in flights.select("FlightDate")
                 .filter(F.col("FlightDate").isNotNull())
                 .distinct().orderBy("FlightDate").collect()]
        windows = compute_windows(dates)
        validation_start, test_start = windows["validation"][0], windows["test"][0]
        log(f"Distinct calendar dates: {len(dates)}")
        for split, (start, end) in windows.items():
            note = "  (test outcome fields withheld until after selection is frozen)" if split == "test" else ""
            log(f"{split} dates: {start} through {end}{note}")

        # All label-based selection accounting is restricted to the pre-test scope.
        # In --mode all, test labels are not brought into an eligible frame or used
        # for counts/metrics until after DECISIONS_FROZEN=1 below.
        selection_source = flights.filter(F.col("FlightDate") < F.lit(test_start))
        category_rows = selection_source.groupBy(eligibility_category_expr().alias("Category")).count().collect()
        category_counts = {r["Category"]: int(r["count"]) for r in category_rows}
        total_rows = sum(category_counts.values())
        log(f"Rows in pre-test selection scope: {total_rows:,}")
        eligibility_rows = [[c, category_counts.get(c, 0)] for c in ELIGIBILITY_ORDER]
        for name, count in eligibility_rows:
            log(f"  pre-test {name}: {count:,}")
        if sum(r[1] for r in eligibility_rows) != total_rows:
            raise PipelineError("Pre-test eligibility categories do not sum to the input row count.")
        eligible_total = category_counts.get("eligible", 0)
        if eligible_total == 0:
            raise PipelineError("No eligible pre-test flights after applying the eligibility rules.")

        demand = keep(compute_demand_table(selection_source))
        eligible = keep(build_eligible_frame(selection_source, holidays, demand))
        check_rows("pre-test eligible frame (holiday + demand joins)", eligible_total, eligible)

        split_expr = (
            F.when(F.col("FlightDate") < F.lit(validation_start), "train")
            .when(F.col("FlightDate") < F.lit(test_start), "validation")
            .otherwise("test")
        )
        accounting = (
            eligible.withColumn("Split", split_expr).groupBy("Split", "HolidayPeriod")
            .agg(F.count(F.lit(1)).alias("n"), F.sum("label").alias("pos")).collect()
        )
        split_names = ["train", "validation", "test"]
        split_n = {s: 0 for s in split_names}
        split_pos = {s: 0 for s in split_names}
        holiday_rows = []
        seen_in_fit = set()
        for r in sorted(accounting, key=lambda r: (split_names.index(r["Split"]), r["HolidayPeriod"])):
            n, pos = int(r["n"]), int(r["pos"] or 0)
            split_n[r["Split"]] += n
            split_pos[r["Split"]] += pos
            holiday_rows.append([r["Split"], r["HolidayPeriod"], n, pos])
            if r["Split"] in ("train", "validation"):
                seen_in_fit.add(r["HolidayPeriod"])
        eligible_count = sum(split_n.values())  # pre-test eligible rows during selection
        active_splits = ["train", "validation"]
        for split in active_splits:
            n, pos = split_n[split], split_pos[split]
            if n == 0 or pos == 0 or pos == n:
                raise PipelineError(f"Split '{split}' is empty or has only one class (n={n}, positives={pos}).")
            log(f"{split}: {n:,} flights; delayed 15+ min: {pos:,} ({100.0 * pos / n:.2f}%)")
        bad_target = eligible.filter(~F.col("label").isin(0.0, 1.0)).limit(1).count()
        if bad_target:
            raise PipelineError("Target is not binary after the eligibility filter.")
        invalid_times = int(eligible.agg(F.sum("TimeParseInvalid")).first()[0] or 0)
        log(f"Eligible flights in this run: {eligible_count:,}; unparsable scheduled times "
            f"(set to neutral values): {invalid_times:,}")
        train_cats = {r[1] for r in holiday_rows if r[0] == "train"}
        for cat in sorted({r[1] for r in holiday_rows if r[1] != "Regular day"}):
            if cat not in train_cats:
                log(f"NOTE: holiday category '{cat}' has no flights in the TRAINING window; its effect cannot be learned there.")
        if ctx.mode == "all":
            for cat in sorted({r[1] for r in holiday_rows if r[0] == "test" and r[1] != "Regular day"}):
                if cat not in seen_in_fit:
                    log(f"NOTE: test holiday category '{cat}' is never seen before the test window "
                        f"(train + validation); the model can only treat it as an unseen category.")

        train_base = eligible.filter(F.col("FlightDate") < F.lit(validation_start))
        validation_base = eligible.filter(
            (F.col("FlightDate") >= F.lit(validation_start)) & (F.col("FlightDate") < F.lit(test_start)))
        val_n, val_pos = split_n["validation"], split_pos["validation"]
        val_prevalence = val_pos / val_n

        # ---------------- stage A: feature experiments (validation only) ----------------
        stamp("STAGE A: one-factor feature experiments with Logistic Regression (validation only)")
        spec, stage_a, feature_results = select_feature_spec(
            ctx.experiment_names, train_base, validation_base, keep, release)
        log(f"Chosen feature spec: {spec['name']} [{spec_label(spec)}]")
        train_vec = stage_a["fit_vec"]
        val_vec = stage_a["applied"]["validation"]
        experiment_rows = []   # dicts, written to experiment_comparison.csv
        ref_result = feature_results["E0_reference"]

        for name, res in feature_results.items():
            experiment_rows.append({
                "Experiment": name, "Description": res["spec"]["factor"], "Spec": res["spec"],
                "Model": f"LogisticRegression(regParam={EXPERIMENT_LR_REG})", "Role": "feature experiment",
                "ROC": res["roc"], "PR": res["pr"], "F1": res["best_f1"], "Thr": res["best_threshold"],
                "Seconds": res["seconds"], "Selected": "feature spec" if name == spec["name"] else "no",
                "Beats": "reference" if name == "E0_reference" else
                         ("yes" if res["pr"] >= ref_result["pr"] + SELECTION_MARGIN else "no"),
            })

        # ---------------- stage A: model tuning on the chosen features ----------------
        search_rows = []
        best_lr = None
        for reg in LR_REG_GRID:
            started = time.time()
            stamp(f"Logistic Regression candidate regParam={reg}")
            model = fit_logistic(train_vec, reg)
            scored = score_dataframe(model, val_vec, keep)
            roc, pr = auc_pair(scored)
            label = f"regParam={reg}"
            search_rows.append(["LogisticRegression", label, roc, pr])
            experiment_rows.append({"Experiment": "tune_LR", "Description": "L2 strength grid", "Spec": spec,
                                    "Model": f"LogisticRegression({label})", "Role": "model tuning",
                                    "ROC": roc, "PR": pr, "Seconds": time.time() - started})
            log(f"  validation AUC-ROC={roc:.5f} AUC-PR={pr:.5f}")
            if best_lr is None or pr > best_lr["pr"] + 1e-12 or (abs(pr - best_lr["pr"]) <= 1e-12 and reg > best_lr["reg"]):
                if best_lr is not None:
                    release(best_lr["scored"])
                best_lr = {"reg": reg, "pr": pr, "roc": roc, "scored": scored, "label": label,
                           "weighted": False}
            else:
                release(scored)

        # class-weighted LR at the best strength: one extra experiment, kept only if clearly better
        started = time.time()
        pos_weight = balanced_pos_weight(fit_prevalence(train_vec))
        stamp(f"Logistic Regression class-weighted candidate (positive weight {pos_weight:.2f}, regParam={best_lr['reg']})")
        weighted_model = fit_logistic(train_vec, best_lr["reg"], pos_weight)
        weighted_scored = score_dataframe(weighted_model, val_vec, keep)
        w_roc, w_pr = auc_pair(weighted_scored)
        weighted_label = f"regParam={best_lr['reg']},classWeight=balanced"
        search_rows.append(["LogisticRegression", weighted_label, w_roc, w_pr])
        experiment_rows.append({"Experiment": "E7_LR_class_weighted", "Description": "balanced class weights",
                                "Spec": spec, "Model": f"LogisticRegression({weighted_label})",
                                "Role": "model tuning", "ROC": w_roc, "PR": w_pr, "Seconds": time.time() - started})
        log(f"  validation AUC-ROC={w_roc:.5f} AUC-PR={w_pr:.5f} (class-weighted)")
        if w_pr >= best_lr["pr"] + SELECTION_MARGIN:
            release(best_lr["scored"])
            best_lr = {"reg": best_lr["reg"], "pr": w_pr, "roc": w_roc, "scored": weighted_scored,
                       "label": weighted_label, "weighted": True}
        else:
            release(weighted_scored)
        log(f"Selected Logistic Regression: {best_lr['label']} (validation AUC-PR={best_lr['pr']:.5f})")

        best_rf = None
        for config in RF_CONFIGS:
            started = time.time()
            label = f"maxDepth={config['maxDepth']},minInstancesPerNode={config['minInstancesPerNode']},numTrees={RF_NUM_TREES}"
            stamp(f"Random Forest candidate {label}")
            model = fit_forest(train_vec, config)
            scored = score_dataframe(model, val_vec, keep)
            roc, pr = auc_pair(scored)
            search_rows.append(["RandomForest", label, roc, pr])
            experiment_rows.append({"Experiment": "tune_RF", "Description": "small RF grid", "Spec": spec,
                                    "Model": f"RandomForest({label})", "Role": "model tuning",
                                    "ROC": roc, "PR": pr, "Seconds": time.time() - started})
            log(f"  validation AUC-ROC={roc:.5f} AUC-PR={pr:.5f}")
            if best_rf is None or pr > best_rf["pr"] + 1e-12:
                if best_rf is not None:
                    release(best_rf["scored"])
                best_rf = {"config": config, "pr": pr, "roc": roc, "scored": scored, "label": label}
            else:
                release(scored)
        log(f"Selected Random Forest: {best_rf['label']} (validation AUC-PR={best_rf['pr']:.5f})")

        # Extra model families (--extra-models): tuned on validation AUC-PR exactly like LR/RF.
        extras = []
        for key in ctx.extra_models:
            family = EXTRA_MODEL_NAMES[key]
            if key == "gbt":
                best = tune_candidates(
                    family, GBT_CONFIGS, lambda c: f"maxIter={c['maxIter']},maxDepth={c['maxDepth']}",
                    fit_gbt, train_vec, val_vec, keep, release, search_rows, experiment_rows, spec,
                    "extra model grid")
                best["fit"] = fit_gbt
            elif key == "mlp":
                best = tune_candidates(
                    family, MLP_CONFIGS, lambda c: "hidden=" + "-".join(str(h) for h in c["hidden"]),
                    fit_mlp, train_vec, val_vec, keep, release, search_rows, experiment_rows, spec,
                    "extra model grid")
                best["fit"] = fit_mlp
            else:
                best = tune_candidates(
                    family, XGB_CONFIGS,
                    lambda c: (f"n={c['n_estimators']},depth={c['max_depth']},"
                               f"eta={c['learning_rate']},minChild={c['min_child_weight']}"),
                    fit_xgb, train_vec, val_vec, keep, release, search_rows, experiment_rows, spec,
                    "distributed XGBoost grid")
                best["fit"] = fit_xgb
            best["family"] = family
            extras.append(best)

        # ---------------- stage A: thresholds on VALIDATION only ----------------
        stamp("Selecting classification thresholds on VALIDATION data only")
        lr_threshold, lr_threshold_rows = threshold_search(best_lr["scored"], THRESHOLD_CANDIDATES)
        rf_threshold, rf_threshold_rows = threshold_search(best_rf["scored"], THRESHOLD_CANDIDATES)
        thresholds = {"LogisticRegression": lr_threshold, "RandomForest": rf_threshold}
        threshold_tables = {"LogisticRegression": lr_threshold_rows, "RandomForest": rf_threshold_rows}
        for e in extras:
            e["threshold"], e["threshold_rows"] = threshold_search(e["scored"], THRESHOLD_CANDIDATES)
            thresholds[e["family"]] = e["threshold"]
            threshold_tables[e["family"]] = e["threshold_rows"]
            log(f"{e['family']} threshold={e['threshold']:.2f} "
                f"(validation F1={max(r[6] for r in e['threshold_rows']):.5f})")
        val_f1 = {m: max(r[6] for r in rows) for m, rows in threshold_tables.items()}
        trivial_val_f1 = 2.0 * val_prevalence / (1.0 + val_prevalence)
        log(f"Logistic Regression threshold={lr_threshold:.2f} (validation F1={val_f1['LogisticRegression']:.5f}); "
            f"Random Forest threshold={rf_threshold:.2f} (validation F1={val_f1['RandomForest']:.5f}); "
            f"trivial 'flag every flight' validation F1={trivial_val_f1:.5f}")

        validation_metric_rows = []
        val_pred_rate = {}
        validation_candidates = [
            ("LogisticRegression", lr_threshold, lr_threshold_rows, best_lr["scored"], best_lr["label"]),
            ("RandomForest", rf_threshold, rf_threshold_rows, best_rf["scored"], best_rf["label"]),
        ] + [(e["family"], e["threshold"], e["threshold_rows"], e["scored"], e["label"]) for e in extras]
        for model_name, thr, rows_, chosen_scored, label in validation_candidates:
            chosen = next(r for r in rows_ if r[0] == thr)
            tp, fp, fn = chosen[1], chosen[2], chosen[3]
            tn = val_n - tp - fp - fn
            precision, recall, f1, accuracy = prf(tp, fp, fn, tn)
            roc, pr = auc_pair(chosen_scored)
            val_pred_rate[model_name] = (tp + fp) / val_n
            validation_metric_rows.append([model_name, label, float(thr), roc, pr, f1, precision, recall, accuracy])
            if recall >= 0.999 or thr <= THRESHOLD_CANDIDATES[0] or thr >= THRESHOLD_CANDIDATES[-1]:
                log(f"WARNING: {model_name} threshold {thr:.2f} is degenerate or at the grid edge "
                    f"(validation recall={recall:.3f}); interpret its F1 with care.")
            if f1 <= trivial_val_f1 + 0.005:
                log(f"WARNING: {model_name} validation F1 ({f1:.5f}) is within 0.005 of the trivial "
                    f"'flag every flight' F1 ({trivial_val_f1:.5f}); F1 barely discriminates here.")

        # Selection is now complete: every decision below was taken on train/validation data only.
        frozen = {
            "frozen_before_test_scoring": True,
            "mode": ctx.mode,
            "feature_spec": {k: spec[k] for k in SPEC_FACTOR_KEYS} | {"name": spec["name"]},
            "feature_spec_label": spec_label(spec),
            "selection_margin_auc_pr": SELECTION_MARGIN,
            "logistic_regression": {"label": best_lr["label"], "regParam": best_lr["reg"],
                                    "class_weighted": best_lr["weighted"], "maxIter": LR_MAX_ITER},
            "random_forest": {"label": best_rf["label"], **best_rf["config"], "numTrees": RF_NUM_TREES,
                              "maxBins": RF_MAX_BINS, "subsamplingRate": RF_SUBSAMPLING_RATE,
                              "featureSubsetStrategy": RF_FEATURE_SUBSET},
            "extra_models": {e["family"]: {"label": e["label"], "config": e["config"],
                                           "threshold": e["threshold"]} for e in extras},
            "thresholds_from_validation_F1": thresholds,
            "windows": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in windows.items()},
            "seed": SEED,
        }
        frozen_text = json.dumps(frozen, indent=2, sort_keys=True)
        write_local_text(ctx, "frozen_config.json", frozen_text + "\n")
        save_text(spark, ctx, "frozen_config", frozen_text)
        log("DECISIONS_FROZEN=1  (feature spec, hyperparameters and thresholds fixed from validation data)")

        stage_a_prior = stage_a["prior"]
        stage_a_names = stage_a["feature_names"]
        for frame in [best_lr["scored"], best_rf["scored"], train_vec, val_vec] + [e["scored"] for e in extras]:
            release(frame)
        for table in stage_a["tables"].values():
            release(table)
        del stage_a

        # ---------------- only now may all-mode read test outcomes ----------------
        test_eligible = None
        if ctx.mode == "all":
            stamp("DECISIONS_FROZEN=1: now reading test-window outcomes for one final evaluation")
            test_source = flights.filter(F.col("FlightDate") >= F.lit(test_start))

            # Merge test eligibility accounting only after the model/feature/threshold
            # decisions have been frozen. This preserves the full-run output interface.
            test_category_rows = test_source.groupBy(eligibility_category_expr().alias("Category")).count().collect()
            test_category_counts = {r["Category"]: int(r["count"]) for r in test_category_rows}
            for category in ELIGIBILITY_ORDER:
                category_counts[category] = category_counts.get(category, 0) + test_category_counts.get(category, 0)
            total_rows = sum(category_counts.values())
            eligibility_rows = [[c, category_counts.get(c, 0)] for c in ELIGIBILITY_ORDER]
            if sum(r[1] for r in eligibility_rows) != total_rows:
                raise PipelineError("Full-run eligibility categories do not sum to the input row count.")

            # Scheduled demand features use schedule fields only. Build test eligibility
            # and read its target only after the frozen_config has been written.
            test_demand = keep(compute_demand_table(test_source))
            test_eligible = keep(build_eligible_frame(test_source, holidays, test_demand))
            test_eligible_count = test_eligible.count()
            test_pos = int(test_eligible.agg(F.sum("label")).first()[0] or 0)
            test_n = test_eligible_count
            if test_n == 0 or test_pos == 0 or test_pos == test_n:
                raise PipelineError(f"Test split is empty or has only one class (n={test_n}, positives={test_pos}).")
            split_n["test"] = test_n
            split_pos["test"] = test_pos
            eligible_total += test_eligible_count
            eligible_count = sum(split_n.values())
            active_splits = ["train", "validation", "test"]
            invalid_times += int(test_eligible.agg(F.sum("TimeParseInvalid")).first()[0] or 0)

            test_holiday_rows = (test_eligible.groupBy("HolidayPeriod")
                                 .agg(F.count(F.lit(1)).alias("n"), F.sum("label").alias("pos"))
                                 .collect())
            for r in sorted(test_holiday_rows, key=lambda row: row["HolidayPeriod"]):
                holiday_rows.append(["test", r["HolidayPeriod"], int(r["n"]), int(r["pos"] or 0)])
            seen_in_fit = {r[1] for r in holiday_rows if r[0] in ("train", "validation")}
            test_categories = {r[1] for r in holiday_rows if r[0] == "test" and r[1] != "Regular day"}
            for cat in sorted(test_categories):
                if cat not in seen_in_fit:
                    log(f"NOTE: test holiday category '{cat}' is never seen before the test window "
                        "(train + validation); the model can only treat it as an unseen category.")
            log(f"Full cleaned row count (read for final evaluation after freeze): {total_rows:,}")
            log(f"Full eligible flight count (pre-test + test): {eligible_total:,}")
            log(f"test: {test_n:,} flights; delayed 15+ min: {test_pos:,} ({100.0 * test_pos / test_n:.2f}%)")

        # ---------------- results shared by both modes ----------------
        stamp("Saving validation-stage result tables")
        threshold_rows = [["LogisticRegression", *r] for r in lr_threshold_rows] + \
                         [["RandomForest", *r] for r in rf_threshold_rows] + \
                         [[e["family"], *r] for e in extras for r in e["threshold_rows"]]
        save_table(spark, ctx, "validation_threshold_search.csv",
                   ["Model", "Threshold", "TP", "FP", "FN", "Precision_class1", "Recall_class1", "F1_class1"],
                   threshold_rows)
        selected_label = {"LogisticRegression": best_lr["label"], "RandomForest": best_rf["label"]}
        selected_label.update({e["family"]: e["label"] for e in extras})
        save_table(spark, ctx, "hyperparameter_search.csv",
                   ["Model", "Config", "ValidationAUC_ROC", "ValidationAUC_PR", "Selected"],
                   [[r[0], r[1], r[2], r[3], "yes" if r[1] == selected_label[r[0]] else "no"] for r in search_rows])
        save_table(spark, ctx, "validation_metrics.csv",
                   ["Model", "Config", "DecisionThreshold", "AUC_ROC", "AUC_PR", "F1_class1",
                    "Precision_class1", "Recall_class1", "Accuracy"], validation_metric_rows)
        save_table(spark, ctx, "eligibility_counts.csv", ["Category", "Flights"], eligibility_rows)
        save_table(spark, ctx, "holiday_category_counts.csv",
                   ["Split", "HolidayPeriod", "Flights", "Delayed15Plus"], holiday_rows)
        class_rows = []
        for split in split_names:
            if split == "test" and ctx.mode == "select":
                class_rows.append([split, "", "", "", windows[split][0].isoformat(), windows[split][1].isoformat()])
            else:
                class_rows.append([split, split_n[split], split_pos[split], split_n[split] - split_pos[split],
                                   windows[split][0].isoformat(), windows[split][1].isoformat()])
        save_table(spark, ctx, "class_counts.csv",
                   ["Split", "Flights", "Delayed15Plus", "NotDelayed15Plus", "StartDate", "EndDate"], class_rows)
        # the previous script's LR ablation, kept for the report: reference (E0) vs no history (E1)
        if "E1_no_history" in feature_results:
            save_table(spark, ctx, "ablation_validation_metrics.csv",
                       ["Variant", "ValidationAUC_ROC", "ValidationAUC_PR"],
                       [["LogisticRegression with historical-rate features", ref_result["roc"], ref_result["pr"]],
                        ["LogisticRegression without historical-rate features",
                         feature_results["E1_no_history"]["roc"], feature_results["E1_no_history"]["pr"]]])

        final_summary = {}
        extra_lines = []
        if ctx.mode == "select":
            log("MODE select finished: test labels were not used for selection, scoring or tuning.")
        else:
            # ---------------- stage B: refit on train + validation, score held-out test ONCE ----------------
            trainval_base = eligible.filter(F.col("FlightDate") < F.lit(test_start))
            if test_eligible is None:
                raise PipelineError("Internal error: test frame was not created after decisions were frozen.")
            test_base = test_eligible
            stage_b = prepare_stage("STAGE B", spec, trainval_base, {"test": test_base}, keep, release)
            final_vec = stage_b["fit_vec"]
            test_vec = stage_b["applied"]["test"]
            stamp(f"Refitting Logistic Regression ({best_lr['label']}) on train + validation")
            lr_pos_weight = balanced_pos_weight(fit_prevalence(final_vec)) if best_lr["weighted"] else None
            lr_final = fit_logistic(final_vec, best_lr["reg"], lr_pos_weight)
            stamp(f"Refitting Random Forest ({best_rf['label']}) on train + validation")
            rf_final = fit_forest(final_vec, best_rf["config"])
            extra_finals = []
            for e in extras:
                stamp(f"Refitting {e['family']} ({e['label']}) on train + validation")
                extra_finals.append((e["family"], e["fit"](final_vec, e["config"]), e["threshold"]))

            test_n, test_pos = split_n["test"], split_pos["test"]
            test_prevalence = test_pos / test_n
            metrics_rows, confusion_rows, ranking_rows, slice_rows, transfer_rows = [], [], [], [], []
            for model_name, model, threshold in [("LogisticRegression", lr_final, lr_threshold),
                                                 ("RandomForest", rf_final, rf_threshold)] + extra_finals:
                stamp(f"Scoring held-out test window: {model_name} (threshold {threshold:.2f} from validation)")
                scored = score_dataframe(model, test_vec, keep)
                roc, pr = auc_pair(scored)
                tp, fp, fn, tn = confusion(scored, threshold)
                if tp + fp + fn + tn != test_n:
                    raise PipelineError(f"{model_name}: confusion counts do not sum to the test row count.")
                precision, recall, f1, accuracy = prf(tp, fp, fn, tn)
                metrics_rows.append([model_name, float(threshold), roc, pr, f1, precision, recall, accuracy])
                final_summary[model_name] = {"roc": roc, "pr": pr, "f1": f1}
                for actual, predicted, count in ((0, 0, tn), (0, 1, fp), (1, 0, fn), (1, 1, tp)):
                    confusion_rows.append([model_name, actual, predicted, count])
                log(f"{model_name} TEST: threshold={threshold:.2f} AUC_ROC={roc:.5f} AUC_PR={pr:.5f} "
                    f"F1_class1={f1:.5f} precision_class1={precision:.5f} recall_class1={recall:.5f} accuracy={accuracy:.5f}")
                transfer_rows.append([model_name, float(threshold), val_prevalence, val_pred_rate[model_name],
                                      test_prevalence, (tp + fp) / test_n])
                for fraction, flagged_fraction, flagged_n, p_, r_, lift in ranking_metrics(scored, TOP_FRACTIONS, test_n, test_pos):
                    ranking_rows.append([model_name, fraction, flagged_fraction, flagged_n, p_, r_, lift])
                slice_rows.append([model_name, "ALL TEST", test_n, test_pos, test_prevalence, roc, pr,
                                   float(threshold), precision, recall, f1, accuracy])
                for period in sorted(r[0] for r in scored.select("HolidayPeriod").distinct().collect()):
                    part = scored.filter(F.col("HolidayPeriod") == period)
                    n_part = part.count()
                    pos_part = int(part.agg(F.sum("label")).first()[0] or 0)
                    roc_part, pr_part = auc_pair(part) if 0 < pos_part < n_part else (None, None)
                    tp_s, fp_s, fn_s, tn_s = confusion(part, threshold)
                    p_s, r_s, f_s, a_s = prf(tp_s, fp_s, fn_s, tn_s)
                    slice_rows.append([model_name, period, n_part, pos_part,
                                       pos_part / n_part if n_part else None, roc_part, pr_part,
                                       float(threshold), p_s, r_s, f_s, a_s])
                release(scored)

            baseline_rows = [
                ["MajorityClass_AlwaysNotDelayed", None, 0.5, test_prevalence, 0.0, 0.0, 0.0, 1.0 - test_prevalence],
                ["AlwaysDelayed", None, 0.5, test_prevalence,
                 2.0 * test_prevalence / (1.0 + test_prevalence), test_prevalence, 1.0, test_prevalence],
            ]
            names = stage_b["feature_names"]
            importances = rf_final.featureImportances.toArray().tolist()
            importance_rows = sorted(
                [[names[i] if i < len(names) else f"feature_{i}", float(v)] for i, v in enumerate(importances)],
                key=lambda r: r[1], reverse=True)[:30]

            stamp("Saving test-stage result tables")
            save_table(spark, ctx, "metrics.csv",
                       ["Model", "DecisionThreshold", "AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1",
                        "Recall_class1", "Accuracy"], metrics_rows)
            save_table(spark, ctx, "confusion_counts.csv",
                       ["Model", "ActualLabel", "PredictedLabel", "Count"], confusion_rows)
            save_table(spark, ctx, "random_forest_feature_importance_top30.csv", ["Feature", "Importance"], importance_rows)
            save_table(spark, ctx, "baseline_metrics.csv",
                       ["Model", "DecisionThreshold", "AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1",
                        "Recall_class1", "Accuracy"], baseline_rows)
            save_table(spark, ctx, "ranking_metrics.csv",
                       ["Model", "TargetFlaggedFraction", "ActualFlaggedFraction", "FlaggedFlights",
                        "Precision_class1", "Recall_class1", "Lift"], ranking_rows)
            save_table(spark, ctx, "metrics_by_holiday_slice.csv",
                       ["Model", "Slice", "Flights", "Delayed15Plus", "Prevalence", "AUC_ROC", "AUC_PR",
                        "DecisionThreshold", "Precision_class1", "Recall_class1", "F1_class1", "Accuracy"], slice_rows)
            save_table(spark, ctx, "threshold_transfer.csv",
                       ["Model", "Threshold", "ValidationPrevalence", "ValidationPredictedPositiveRate",
                        "TestPrevalence", "TestPredictedPositiveRate"], transfer_rows)

            stamp("Saving fitted preprocessing pipeline, classifiers and historical-rate tables")
            stage_b["pipeline_model"].write().overwrite().save(f"{ctx.hdfs_models}/feature_pipeline")
            lr_final.write().overwrite().save(f"{ctx.hdfs_models}/logistic_regression")
            rf_final.write().overwrite().save(f"{ctx.hdfs_models}/random_forest")
            for extra_name, extra_model, _ in extra_finals:
                extra_model.write().overwrite().save(f"{ctx.hdfs_models}/{MODEL_DIRS[extra_name]}")
            for table_name, table in stage_b["tables"].items():
                table.write.mode("overwrite").parquet(f"{ctx.hdfs_models}/historical_rate_tables/{table_name}")
            metadata = dict(frozen)
            metadata.update({
                "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "spark_version": spark.version,
                "target": "ArrDel15 (1 = arrival delay >= 15 minutes); cancelled/diverted excluded",
                "final_fit_window": [windows["train"][0].isoformat(), windows["validation"][1].isoformat()],
                "feature_source_columns": FEATURE_SOURCE_COLUMNS,
                "features": names, "features_stage_a": stage_a_names,
                "history_groups": {g["name"]: {"keys": g["keys_map"], "window_days": g["window_days"]}
                                   for g in history_groups(spec)},
                "history_smoothing": HISTORY_SMOOTHING,
                "history_prior_stage_a_train": stage_a_prior, "history_prior_final_fit": stage_b["prior"],
            })
            metadata_text = json.dumps(metadata, indent=2)
            write_local_text(ctx, "preprocessing_metadata.json", metadata_text + "\n")
            save_text(spark, ctx, "preprocessing_metadata", metadata_text)
            log(f"Saved models under {ctx.hdfs_models}")
            extra_lines = [f"{r[0]}: threshold transfer validation predicted-positive rate={r[3]:.4f} "
                           f"vs test={r[5]:.4f} (prevalence {r[2]:.4f} -> {r[4]:.4f})" for r in transfer_rows]

        # ---------------- experiment comparison table ----------------
        header = ["Experiment", "Description", "UsesHistory", "HistoryWindowDays", "WarmupDays",
                  "UsesHolidayFeatures", "UsesScheduledDemand", "HistorySet", "RecentWindowDays",
                  "UsesExtraDemand", "UsesScheduleExtras", "Model", "Role",
                  "ValidationAUC_ROC", "ValidationAUC_PR", "ValidationBestF1", "ValidationBestThreshold",
                  "DeltaAUC_PR_vs_Reference", "BeatsReferenceByMargin", "Selected", "Seconds",
                  "TestAUC_PR", "TestF1"]
        table_rows = []
        for e in experiment_rows:
            model_family = e["Model"].split("(")[0]
            is_final = (e["Role"] == "model tuning" and e["Model"].split("(", 1)[-1].rstrip(")") == selected_label.get(model_family))
            test_pr = test_f1 = None
            if is_final and model_family in final_summary:
                test_pr, test_f1 = final_summary[model_family]["pr"], final_summary[model_family]["f1"]
            table_rows.append([
                e["Experiment"], e["Description"], *tidy_spec_fields(e["Spec"]), e["Model"], e["Role"],
                e["ROC"], e["PR"], e.get("F1"), e.get("Thr"), e["PR"] - ref_result["pr"],
                e.get("Beats", ""), e.get("Selected", "final candidate" if is_final else "no"),
                round(e["Seconds"], 1), test_pr, test_f1])
        save_table(spark, ctx, "experiment_comparison.csv", header, table_rows)

        # ---------------- run summary ----------------
        summary_lines = [
            "BDA MLlib flight-arrival-delay classification (redesigned ML stage)",
            f"Mode: {ctx.mode}; namespace: {ctx.namespace}",
            "Target: ArrDel15; positive class: arrival delay >= 15 minutes",
            "Cancelled and diverted flights excluded; valid 0/1 target required.",
            "Predictors: pre-departure schedule, carrier, cyclic time features, holiday calendar, optional",
            "  scheduled origin-hour demand and smoothed historical delay rates (strictly earlier dates).",
            "No post-departure or delay-cause columns, no Tail_Number, no raw Year/Month/DayOfYear.",
            f"Input rows: {total_rows}; eligible rows in input: {eligible_total}",
            f"Distinct dates: {len(dates)}",
            f"Training date range: {windows['train'][0]} through {windows['train'][1]}",
            f"Validation date range (tuning and threshold selection): {windows['validation'][0]} through {windows['validation'][1]}",
        ]
        if ctx.mode == "all":
            summary_lines.append(f"Final test date range (held out, scored once): {windows['test'][0]} through {windows['test'][1]}")
        else:
            summary_lines.append("Test labels/eligibility: not used in --mode select; FlightDate only is read to define the cutoff.")
        for split in active_splits:
            summary_lines.append(f"{split.capitalize()} records: {split_n[split]}; positive: {split_pos[split]} "
                                 f"({100.0 * split_pos[split] / split_n[split]:.2f}%)")
        summary_lines += [
            f"Chosen feature spec: {spec['name']} [{spec_label(spec)}] (margin {SELECTION_MARGIN} AUC-PR over E0)",
            f"Selected Logistic Regression: {best_lr['label']} (maxIter={LR_MAX_ITER})",
            f"Selected Random Forest: {best_rf['label']} (maxBins={RF_MAX_BINS}, subsamplingRate={RF_SUBSAMPLING_RATE}, "
            f"featureSubsetStrategy={RF_FEATURE_SUBSET})",
            *[f"Selected extra model {e['family']}: {e['label']}" for e in extras],
            "Hyperparameters chosen by validation AUC-PR; thresholds chosen by validation F1 only.",
            "AUC-ROC and AUC-PR use the class-1 probability; F1, precision and recall refer to class 1",
            "at the validation-selected threshold. Holiday categories absent from the fitting windows are",
            "treated as unseen categories. Results are observational, not causal.", "",
            "Validation metrics at the chosen thresholds:",
            "Model, Config, DecisionThreshold, AUC_ROC, AUC_PR, F1_class1, Precision_class1, Recall_class1, Accuracy",
        ]
        for row in validation_metric_rows:
            summary_lines.append(", ".join([row[0], f'"{row[1]}"'] + [f"{v:.6f}" for v in row[2:]]))
        if ctx.mode == "all":
            summary_lines += [
                "", f"Trivial references on test: majority class accuracy={1.0 - test_prevalence:.4f}; "
                    f"'flag every flight' F1={2.0 * test_prevalence / (1.0 + test_prevalence):.4f}; "
                    f"random-ranking AUC-PR equals prevalence={test_prevalence:.4f}.",
                "Held-out test metrics (single evaluation after decisions were frozen):",
                "Model, DecisionThreshold, AUC_ROC, AUC_PR, F1_class1, Precision_class1, Recall_class1, Accuracy"]
            for row in metrics_rows:
                summary_lines.append(", ".join([row[0]] + [f"{value:.6f}" for value in row[1:]]))
            summary_lines += extra_lines
        summary_text = "\n".join(summary_lines) + "\n"
        write_local_text(ctx, "run_summary.txt", summary_text)
        log(f"Local run summary: {os.path.join(ctx.local_results, 'run_summary.txt')}")

        log(f"ML_ELIGIBLE_ROWS={eligible_total}")
        log(f"ML_TRAIN_ROWS={split_n['train']}")
        log(f"ML_VALIDATION_ROWS={split_n['validation']}")
        if ctx.mode == "all":
            log(f"ML_TEST_ROWS={split_n['test']}")
            log(f"ML_FINAL_FIT_ROWS={split_n['train'] + split_n['validation']}")
            log(f"ML_TEST_START_DATE={test_start.isoformat()}")
        log("MLLIB_JOB_STATUS=SUCCEEDED")
        log(f"MLLIB_FINISHED_UTC={datetime.utcnow().isoformat(timespec='seconds')}Z")
        return 0
    except PipelineError as error:
        log(f"ERROR: {error}")
        log("MLLIB_JOB_STATUS=FAILED")
        return 2
    except Exception as error:
        log(f"ERROR: MLlib pipeline failed: {type(error).__name__}: {error}")
        log(traceback.format_exc())
        log("MLLIB_JOB_STATUS=FAILED")
        return 1
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist()
            except Exception:
                pass
        if owns_spark and spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
