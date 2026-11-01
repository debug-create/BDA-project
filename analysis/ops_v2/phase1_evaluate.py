#!/usr/bin/env python3
"""Bootstrap, rolling-origin, and gate evaluation for Phase 1 (never reads test)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import host_common as common
import phase1_host as phase1

OUT = common.RESULTS


def json_write(name, value):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def subset(data, ids):
    result = {}
    n = len(data["label"])
    for key, value in data.items():
        if key == "categories_raw":
            result[key] = {name: column.iloc[ids].reset_index(drop=True) for name, column in value.items()}
        elif isinstance(value, np.ndarray) and len(value) == n:
            result[key] = value[ids]
        elif isinstance(value, list) and len(value) == n:
            result[key] = [value[index] for index in ids]
        else:
            result[key] = value
    result["fit_eligible"] = np.ones(len(ids), dtype=bool)
    return result


def top10(y, score):
    row = common.metrics(y, score, fractions=(.10,))["ranking"][0]
    return {"precision": row["precision"], "lift": row["lift"]}


def fold_scores(model_name, config, fit_data, holdout):
    x_fit, names = common.phase1_matrix(fit_data); x_hold, _ = common.phase1_matrix(holdout)
    x_fit, names = phase1.add_interactions(x_fit, names, fit_data)
    x_hold, _ = phase1.add_interactions(x_hold, common.phase1_matrix(holdout)[1], holdout)
    cat_fit, cat_hold, _ = common.encode_categories(fit_data, holdout)
    lr_score, _ = phase1.fit_interaction_lr(fit_data, holdout, x_fit, x_hold, cat_fit, cat_hold)
    all_fit = np.column_stack([x_fit, cat_fit]).astype(np.float32)
    all_hold = np.column_stack([x_hold, cat_hold]).astype(np.float32)
    if model_name == "lightgbm":
        _, candidate, _ = phase1.fit_lgb(config, all_fit, fit_data["label"], all_hold,
                                          holdout["label"], cat_fit.shape[1])
    elif model_name == "xgboost":
        _, candidate, _ = phase1.fit_xgb(config, all_fit, fit_data["label"], all_hold,
                                          holdout["label"], cat_fit.shape[1])
    else:
        _, candidate, _ = phase1.fit_cat(config, x_fit, cat_fit, fit_data["label"],
                                          x_hold, cat_hold, holdout["label"])
    return lr_score, candidate


def main():
    metadata = common.load_metadata(); validation = common.load_split("validation", metadata)
    reference = np.load(OUT / "frozen_utc_lr_validation_scores.npy") if (OUT / "frozen_utc_lr_validation_scores.npy").exists() else common.sigmoid(validation["lr_offset"])
    baseline = json.loads((OUT / "baseline_results.json").read_text(encoding="utf-8"))
    models = {}
    for name in ["lightgbm", "xgboost", "catboost"]:
        summary_path = OUT / f"{name}_best.json"
        if summary_path.exists():
            score = np.load(OUT / f"{name}_validation_scores.npy")
            measured = common.metrics(validation["label"], score)
            models[name] = {
                "metrics": measured,
                "bootstrap_vs_frozen_utc_lr": common.date_block_bootstrap(
                    validation["date"], validation["label"], score, reference, replicates=1000),
                "config": json.loads(summary_path.read_text(encoding="utf-8"))["config"],
            }
    if not models: raise RuntimeError("no completed boosted model searches")
    winner = max(models, key=lambda name: models[name]["metrics"]["pr_auc"])
    config = models[winner]["config"]
    train = common.load_split("train", metadata)
    folds = []
    for start_text in ["2024-10-28", "2024-11-04", "2024-11-11", "2024-11-18"]:
        start = np.datetime64(start_text); stop = start + np.timedelta64(7, "D")
        fit_ids = np.flatnonzero(train["date"] < start)
        hold_ids = np.flatnonzero((train["date"] >= start) & (train["date"] < stop))
        fit_data = subset(train, fit_ids); holdout = subset(train, hold_ids)
        lr_score, candidate = fold_scores(winner, config, fit_data, holdout)
        lr_metrics = common.metrics(holdout["label"], lr_score)
        candidate_metrics = common.metrics(holdout["label"], candidate)
        folds.append({
            "fit_end": str(start - np.timedelta64(1, "D")), "validation_start": start_text,
            "validation_end": str(stop - np.timedelta64(1, "D")), "fit_rows": len(fit_ids),
            "validation_rows": len(hold_ids), "lr_pr_auc": lr_metrics["pr_auc"],
            "candidate_pr_auc": candidate_metrics["pr_auc"],
            "pr_auc_gain": candidate_metrics["pr_auc"] - lr_metrics["pr_auc"],
            "lr_top10": top10(holdout["label"], lr_score),
            "candidate_top10": top10(holdout["label"], candidate),
        })
        json_write("phase1_folds.json", folds)
        print("FOLD=" + json.dumps(folds[-1]), flush=True)
    gains = np.asarray([row["pr_auc_gain"] for row in folds])
    bootstrap = models[winner]["bootstrap_vs_frozen_utc_lr"]
    validation_gain = models[winner]["metrics"]["pr_auc"] - baseline["frozen_utc_lr"]["pr_auc"]
    passed = bool(gains.mean() >= .003 and bootstrap["ci_95_low"] > 0 and validation_gain >= .002)
    decision = {
        "winner": winner, "validation_gain_vs_frozen_utc_lr": validation_gain,
        "rolling_gain_mean": float(gains.mean()), "rolling_gain_std": float(gains.std(ddof=1)),
        "validation_bootstrap": bootstrap, "selection_margin_met": validation_gain >= .002,
        "decision_rule_passed": passed,
        "test_authorized_by_phase1_alone": passed,
        "test_opened": False,
    }
    json_write("phase1_validation_summary.json", {"baseline": baseline, "models": models})
    json_write("phase1_decision.json", decision)
    print(json.dumps(decision, sort_keys=True)); print("PHASE1_EVALUATION_STATUS=SUCCEEDED")


if __name__ == "__main__":
    main()
