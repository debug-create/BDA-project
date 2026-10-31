#!/usr/bin/env python3
"""Paired date-block bootstrap for operations-aware minus schedule-only PR-AUC.

Both designs are represented by Spark-produced per-date, 1,000-score-bin
histograms.  Whole test dates are sampled with replacement and the same sampled
dates are used for both designs, preserving the paired comparison.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/experiments/ml_redesign")
SCHEDULE_HIST = ROOT / "diagnostics/score_histograms_1000bin"
OPS_ROOT = ROOT / "asof_ops/final120"
N_BOOT = 1000
SEED = 42
STATUS = "SECOND DESIGN EVALUATION - test used once after validation freeze"


def average_precision(hist):
    total_pos = sum(positive for _, positive in hist.values())
    if total_pos <= 0:
        return np.nan
    cumulative_n = 0
    cumulative_pos = 0
    result = 0.0
    for score_bin in sorted(hist, reverse=True):
        count, positive = hist[score_bin]
        cumulative_n += count
        cumulative_pos += positive
        result += (positive / total_pos) * (cumulative_pos / cumulative_n)
    return result


def read_schedule():
    data = defaultdict(dict)
    parts = sorted(SCHEDULE_HIST.glob("part-*.csv"))
    if not parts:
        raise SystemExit(f"No schedule-only histogram found under {SCHEDULE_HIST}")
    for part in parts:
        with part.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("Split") != "test":
                    continue
                key = (row["Model"], row["FlightDate"])
                data[key][int(row["ScoreBin"])] = (int(row["Flights"]), int(row["Delayed15Plus"]))
    return data


def read_ops(model, path):
    data = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # hdfs -getmerge concatenates one header per Spark part.
            if row.get("FlightDate") == "FlightDate":
                continue
            key = (model, row["FlightDate"])
            data[key][int(row["ScoreBin"])] = (int(row["Flights"]), int(row["Delayed15Plus"]))
    return data


def aggregate(data, model, sampled_dates):
    result = defaultdict(lambda: [0, 0])
    for day in sampled_dates:
        for score_bin, (count, positive) in data[(model, day)].items():
            result[score_bin][0] += count
            result[score_bin][1] += positive
    return result


def main():
    schedule = read_schedule()
    operations = {}
    operations.update(read_ops("LogisticRegression", OPS_ROOT / "lr_score_histograms_1000bin.csv"))
    operations.update(read_ops("RandomForest", OPS_ROOT / "rf_score_histograms_1000bin.csv"))
    rng = np.random.default_rng(SEED)
    rows = []
    for model in ("LogisticRegression", "RandomForest"):
        schedule_dates = {day for name, day in schedule if name == model}
        operations_dates = {day for name, day in operations if name == model}
        if schedule_dates != operations_dates:
            raise SystemExit(f"Date mismatch for {model}: schedule={schedule_dates}, operations={operations_dates}")
        dates = sorted(schedule_dates)
        schedule_point = average_precision(aggregate(schedule, model, dates))
        operations_point = average_precision(aggregate(operations, model, dates))
        deltas = []
        for _ in range(N_BOOT):
            sampled = rng.choice(dates, size=len(dates), replace=True).tolist()
            schedule_ap = average_precision(aggregate(schedule, model, sampled))
            operations_ap = average_precision(aggregate(operations, model, sampled))
            deltas.append(operations_ap - schedule_ap)
        low, high = np.percentile(np.asarray(deltas), [2.5, 97.5])
        rows.append([STATUS, model, schedule_point, operations_point,
                     operations_point - schedule_point, float(low), float(high),
                     N_BOOT, len(dates), "paired date-block bootstrap; 1000 score bins; seed 42"])

    output = OPS_ROOT / "bootstrap_improvement.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Status", "Model", "ScheduleOnlyAUC_PR_Histogram",
                         "OperationsAwareT120AUC_PR_Histogram", "ImprovementAUC_PR",
                         "CI95_Lower", "CI95_Upper", "BootstrapResamples", "DateBlocks", "Method"])
        writer.writerows(rows)
    print(f"ASOF_OPS_BOOTSTRAP_STATUS=SUCCEEDED output={output} resamples={N_BOOT}")


if __name__ == "__main__":
    main()
