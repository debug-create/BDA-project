#!/usr/bin/env python3
"""Paired date-block bootstrap CIs for the UTC validation ablation."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def area_under_pr(bins):
    """Trapezoidal PR area from (score_bin, rows, positives) aggregates."""
    total_positive = sum(positive for _, _, positive in bins)
    if not total_positive:
        return float("nan")
    tp = fp = 0
    previous_recall, previous_precision = 0.0, 1.0
    area = 0.0
    for _, rows, positive in sorted(bins, reverse=True):
        tp += positive
        fp += rows - positive
        recall = tp / total_positive
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * (precision + previous_precision) / 2.0
        previous_recall, previous_precision = recall, precision
    return area


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()

    by_variant_date = defaultdict(lambda: defaultdict(list))
    with open(args.input, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_variant_date[row["Variant"]][row["FlightDate"]].append(
                (int(row["ScoreBin"]), int(row["Flights"]), int(row["Delayed15Plus"])))

    reference = "a_origin_only_utc"
    comparisons = ["d_all_new_utc", "c_all_old_local_clock"]
    dates = sorted(by_variant_date[reference])
    if not dates or any(sorted(by_variant_date[name]) != dates for name in comparisons):
        raise RuntimeError("variants do not contain the same non-empty validation date blocks")

    rng = random.Random(args.seed)
    differences = {name: [] for name in comparisons}
    for _ in range(args.replicates):
        sampled = [rng.choice(dates) for _ in dates]
        reference_auc = area_under_pr(
            [item for date in sampled for item in by_variant_date[reference][date]])
        for name in comparisons:
            candidate_auc = area_under_pr(
                [item for date in sampled for item in by_variant_date[name][date]])
            differences[name].append(candidate_auc - reference_auc)

    result = {
        "method": "paired nonparametric date-block bootstrap; 1000-bin validation scores; percentile CI",
        "replicates": args.replicates,
        "seed": args.seed,
        "validation_date_blocks": len(dates),
        "comparisons": {},
    }
    for name, values in differences.items():
        result["comparisons"][f"{name}_minus_{reference}"] = {
            "mean_difference": sum(values) / len(values),
            "ci_95_low": percentile(values, 0.025),
            "ci_95_high": percentile(values, 0.975),
            "probability_difference_gt_0": sum(value > 0 for value in values) / len(values),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
