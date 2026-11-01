#!/usr/bin/env python3
"""Validation-only on/off factors for the additional host features."""

import json

import numpy as np

import host_common as common
import phase1_host as phase1


def fit(name, config, train_x, train_cat, y, val_x, val_cat, val_y):
    if name == "lightgbm":
        _, score, iteration = phase1.fit_lgb(config, np.column_stack([train_x, train_cat]), y,
                                              np.column_stack([val_x, val_cat]), val_y, train_cat.shape[1])
    elif name == "xgboost":
        _, score, iteration = phase1.fit_xgb(config, np.column_stack([train_x, train_cat]), y,
                                              np.column_stack([val_x, val_cat]), val_y, train_cat.shape[1])
    else:
        _, score, iteration = phase1.fit_cat(config, train_x, train_cat, y, val_x, val_cat, val_y)
    return score, iteration


def main():
    out = common.RESULTS; metadata = common.load_metadata()
    train = common.load_split("train", metadata); val = common.load_split("validation", metadata)
    available = []
    for name in ["lightgbm", "xgboost", "catboost"]:
        path = out / f"{name}_best.json"
        if path.exists(): available.append((json.loads(path.read_text(encoding="utf-8"))["best_validation"]["pr_auc"], name))
    if not available: raise RuntimeError("boosted searches have not completed")
    model_name = max(available)[1]
    best = json.loads((out / f"{model_name}_best.json").read_text(encoding="utf-8")); config = best["config"]
    train_x, names = common.phase1_matrix(train); val_x, _ = common.phase1_matrix(val)
    train_x, names = phase1.add_interactions(train_x, names, train)
    val_x, _ = phase1.add_interactions(val_x, common.phase1_matrix(val)[1], val)
    train_cat, val_cat, _ = common.encode_categories(train, val); eligible = train["fit_eligible"]
    factors = {
        "explicit_interactions": [index for index, name in enumerate(names) if name.startswith("interaction_")],
        "one_hour_windows": [index for index, name in enumerate(names) if "1h" in name],
        "trend_features": [index for index, name in enumerate(names) if "Trend" in name or "trend" in name],
        "missing_count_masks": [index for index, name in enumerate(names) if name.endswith(("__mask", "__count"))],
        "origin_congestion": [index for index, name in enumerate(names) if name == "OriginCongestionRatio"],
        "route_earlier_today": [index for index, name in enumerate(names) if name.startswith("RouteEarlierToday")],
        "six_hour_state_summaries": [index for index, name in enumerate(names) if name.startswith("seq_")],
    }
    full_score = np.load(out / f"{model_name}_validation_scores.npy")
    full_pr = common.metrics(val["label"], full_score)["pr_auc"]; rows = []
    for factor, removed in factors.items():
        keep = np.ones(len(names), dtype=bool); keep[removed] = False
        score, iteration = fit(model_name, config, train_x[eligible][:, keep], train_cat[eligible],
                               train["label"][eligible], val_x[:, keep], val_cat, val["label"])
        measured = common.metrics(val["label"], score)
        row = {"factor_removed": factor, "removed_columns": len(removed), "pr_auc": measured["pr_auc"],
               "full_pr_auc": full_pr, "full_minus_without": full_pr - measured["pr_auc"],
               "best_iteration": iteration}
        rows.append(row)
        (out / "phase1_factor_ablation.json").write_text(json.dumps(
            {"model": model_name, "full_pr_auc": full_pr, "rows": rows}, indent=2) + "\n", encoding="utf-8")
        print("FACTOR=" + json.dumps(row), flush=True)
    print("PHASE1_FACTOR_ABLATION_STATUS=SUCCEEDED")


if __name__ == "__main__":
    main()
