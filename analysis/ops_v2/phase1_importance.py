#!/usr/bin/env python3
"""Train-gap and grouped permutation importance for the selected Phase-1 model."""

import json
import pickle

import numpy as np
from sklearn.metrics import average_precision_score

import host_common as common
import phase1_host as phase1


def predict(name, model, numeric, categories):
    if name == "lightgbm":
        return model.predict(np.column_stack([numeric, categories]).astype(np.float32))
    if name == "xgboost":
        import xgboost as xgb
        matrix = np.column_stack([numeric, categories]).astype(np.float32)
        types = ["q"] * numeric.shape[1] + ["c"] * categories.shape[1]
        return model.predict(xgb.DMatrix(matrix, feature_types=types, enable_categorical=True))
    from catboost import FeaturesData, Pool
    data = FeaturesData(num_feature_data=numeric.astype(np.float32, copy=False),
                        cat_feature_data=phase1.cat_object(categories))
    return model.predict_proba(Pool(data))[:, 1]


def main():
    out = common.RESULTS
    decision = json.loads((out / "phase1_decision.json").read_text(encoding="utf-8"))
    name = decision["winner"]
    with (out / f"{name}_best_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    metadata = common.load_metadata(); train = common.load_split("train", metadata); val = common.load_split("validation", metadata)
    train_x, names = common.phase1_matrix(train); val_x, _ = common.phase1_matrix(val)
    train_x, names = phase1.add_interactions(train_x, names, train)
    val_x, _ = phase1.add_interactions(val_x, common.phase1_matrix(val)[1], val)
    train_cat, val_cat, _ = common.encode_categories(train, val)
    fit = train["fit_eligible"]
    base = predict(name, model, val_x, val_cat)
    baseline_pr = float(average_precision_score(val["label"], base))
    train_score = predict(name, model, train_x[fit], train_cat[fit])
    train_pr = float(average_precision_score(train["label"][fit], train_score))
    group_for = {}
    for index, feature in enumerate(names):
        if feature.startswith(("Ops", "seq_", "RouteEarlier")):
            group = "operations"
        elif feature.startswith(("hist_", "recent_")):
            group = "history"
        elif "Demand" in feature or feature == "OriginCongestionRatio":
            group = "demand"
        elif feature.startswith("interaction_"):
            group = "explicit_interactions"
        else:
            group = "schedule"
        group_for.setdefault(group, []).append(index)
    rng = np.random.default_rng(20261101); importance = []
    for group in sorted(group_for) + ["categoricals"]:
        drops = []
        for _ in range(5):
            permutation = rng.permutation(len(val["label"]))
            changed_x = val_x.copy() if group != "categoricals" else val_x
            changed_cat = val_cat.copy() if group == "categoricals" else val_cat
            if group == "categoricals":
                changed_cat[:] = changed_cat[permutation]
            else:
                columns = group_for[group]
                changed_x[:, columns] = changed_x[permutation][:, columns]
            score = predict(name, model, changed_x, changed_cat)
            drops.append(baseline_pr - float(average_precision_score(val["label"], score)))
        importance.append({"group": group, "pr_auc_drop_mean": float(np.mean(drops)),
                           "pr_auc_drop_std": float(np.std(drops, ddof=1)), "repeats": 5})
    result = {"model": name, "train_pr_auc": train_pr, "validation_pr_auc": baseline_pr,
              "overfit_gap": train_pr - baseline_pr, "group_permutation_importance": importance}
    (out / "phase1_importance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); print("PHASE1_IMPORTANCE_STATUS=SUCCEEDED")


if __name__ == "__main__":
    main()
