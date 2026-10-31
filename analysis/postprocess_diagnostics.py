#!/usr/bin/env python3
"""Derive small diagnostic summary tables from completed Spark outputs."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "ml_redesign" / "diagnostics"
TEST_STATUS = "TEST - DESCRIPTIVE ONLY, not used for any decision"
VALIDATION_STATUS = "VALIDATION - used for diagnostics/decisions"


def rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


skew = rows("history_feature_skew.csv")
train = {r["Feature"]: r for r in skew if r["Split"] == "train"}
derived = []
for row in skew:
    if row["Split"] == "train":
        continue
    base = train[row["Feature"]]
    std = float(base["StdDev"] or 0.0)
    delta = float(row["Mean"]) - float(base["Mean"])
    standardized = abs(delta) / std if std else None
    derived.append([
        TEST_STATUS if row["Split"] == "test" else VALIDATION_STATUS,
        row["Split"], row["Feature"], base["Mean"], row["Mean"], delta,
        "" if standardized is None else standardized,
    ])
derived.sort(key=lambda r: (r[1], -(float(r[6]) if r[6] != "" else -1)))
with (OUT / "history_feature_shift_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["Status", "Split", "Feature", "TrainingMean", "ServingMean", "MeanShift",
                     "AbsoluteMeanShiftInTrainingStdDevs"])
    writer.writerows(derived)

print("POSTPROCESS_STATUS=SUCCEEDED")
