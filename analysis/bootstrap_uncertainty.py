#!/usr/bin/env python3
"""Date-block bootstrap from Spark-produced per-date, 1000-bin histograms."""

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/experiments/ml_redesign/diagnostics")
N_BOOT = 1000
SEED = 42
VALIDATION_STATUS = "VALIDATION - used for diagnostics/decisions"
TEST_STATUS = "TEST - DESCRIPTIVE ONLY, not used for any decision"


def average_precision(hist):
    total_pos = sum(p for _, p in hist.values())
    if total_pos <= 0:
        return np.nan
    cum_n = 0
    cum_p = 0
    ap = 0.0
    for score_bin in sorted(hist, reverse=True):
        n, p = hist[score_bin]
        cum_n += n
        cum_p += p
        ap += (p / total_pos) * (cum_p / cum_n)
    return ap


def aggregate(data, split, model, sampled_dates):
    out = defaultdict(lambda: [0, 0])
    for day in sampled_dates:
        for score_bin, (n, p) in data[(split, model, day)].items():
            out[score_bin][0] += n
            out[score_bin][1] += p
    return out


def main():
    parts = glob.glob(str(OUT / "score_histograms_1000bin" / "part-*.csv"))
    if not parts:
        raise SystemExit("No Spark histogram part file found; run analysis/model_diagnostics.py first")
    data = defaultdict(dict)
    for part in parts:
        with open(part, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["Split"], row["Model"], row["FlightDate"])
                value = (int(row["Flights"]), int(row["Delayed15Plus"]))
                data[key][int(row["ScoreBin"])] = value

    rng = np.random.default_rng(SEED)
    summary_rows = []
    for split in ("validation", "test"):
        status = TEST_STATUS if split == "test" else VALIDATION_STATUS
        dates = sorted({key[2] for key in data if key[0] == split})
        point_lr = aggregate(data, split, "LogisticRegression", dates)
        point_rf = aggregate(data, split, "RandomForest", dates)
        lr_point = average_precision(point_lr)
        rf_point = average_precision(point_rf)
        prevalence_point = sum(v[1] for v in point_lr.values()) / sum(v[0] for v in point_lr.values())
        stats = {"LR_AUC_PR": [], "RF_AUC_PR": [], "LR_minus_RF": [], "LR_minus_prevalence": []}
        for iteration in range(N_BOOT):
            sampled = rng.choice(dates, size=len(dates), replace=True).tolist()
            lr_hist = aggregate(data, split, "LogisticRegression", sampled)
            rf_hist = aggregate(data, split, "RandomForest", sampled)
            lr = average_precision(lr_hist)
            rf = average_precision(rf_hist)
            prevalence = sum(v[1] for v in lr_hist.values()) / sum(v[0] for v in lr_hist.values())
            values = {"LR_AUC_PR": lr, "RF_AUC_PR": rf, "LR_minus_RF": lr - rf,
                      "LR_minus_prevalence": lr - prevalence}
            for metric, value in values.items():
                stats[metric].append(value)
        points = {"LR_AUC_PR": lr_point, "RF_AUC_PR": rf_point, "LR_minus_RF": lr_point - rf_point,
                  "LR_minus_prevalence": lr_point - prevalence_point}
        for metric, values in stats.items():
            lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
            summary_rows.append([status, split, metric, points[metric], float(lo), float(hi), N_BOOT,
                                 len(dates), "date-block bootstrap; 1000 score bins"])

    with (OUT / "bootstrap_uncertainty.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Status", "Split", "Metric", "PointEstimateFromHistogram", "CI95_Lower", "CI95_Upper",
                         "BootstrapResamples", "DateBlocks", "Method"])
        writer.writerows(summary_rows)
    print(f"BOOTSTRAP_STATUS=SUCCEEDED resamples={N_BOOT}")


if __name__ == "__main__":
    main()
