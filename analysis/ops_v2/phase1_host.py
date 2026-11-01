#!/usr/bin/env python3
"""Phase-1 host baselines and 40-trial boosted-model searches."""

import argparse
import csv
import gc
import json
import os
import pickle
import random
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import host_common as common

SEED = 20261101
TRIALS = 40


def worker_threads():
    return int(os.environ.get("OPS_V2_THREADS", max(1, (psutil.cpu_count(logical=True) or 2) - 1)))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baseline", "search"], required=True)
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--trials", type=int, default=TRIALS)
    return parser.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_scores(name, validation, score):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = common.RESULTS / "scores"; out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"FlightKey": validation["key"], "FlightDate": validation["date"],
                             "label": validation["label"], "score": score.astype(np.float64),
                             "Model": [name] * len(score)}), out / f"{name}.parquet", compression="zstd")


def add_interactions(x, names, data):
    index = {name: i for i, name in enumerate(names)}
    backlog = x[:, index["OpsOriginBacklogShare3h"]]
    hour = data["categories_raw"]["ScheduledDepHour"].astype(float).fillna(0).to_numpy()
    block = np.minimum((hour // 4).astype(int), 5)
    interactions = [(backlog * (block == value)).astype(np.float32) for value in range(6)]
    delay_share = x[:, index["OpsOriginDepDelay15Share3h"]]
    carrier_history = x[:, index["hist_rate_airline"]]
    interactions.append((delay_share * carrier_history).astype(np.float32))
    return np.column_stack([x] + interactions), names + [f"interaction_backlog_depblock_{i}" for i in range(6)] + [
        "interaction_origin_delay_share_x_carrier_history"]


def fit_interaction_lr(train, validation, x_train, x_val, cat_train, cat_val):
    fit = train["fit_eligible"]
    x_train, names = add_interactions(x_train, [], train) if False else (x_train, None)
    # Interactions are added before this function; fit transforms use training only.
    scaler = StandardScaler().fit(x_train[fit])
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    onehot.fit(cat_train[fit])
    a = sparse.hstack([sparse.csr_matrix(scaler.transform(x_train[fit]).astype(np.float32)),
                       onehot.transform(cat_train[fit])], format="csr")
    b = sparse.hstack([sparse.csr_matrix(scaler.transform(x_val).astype(np.float32)),
                       onehot.transform(cat_val)], format="csr")
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, max_iter=40, tol=1e-4,
                          average=True, random_state=SEED, n_jobs=-1)
    model.fit(a, train["label"][fit])
    return model.predict_proba(b)[:, 1], {"solver": "SGD log-loss", "alpha": 1e-6, "epochs_max": 40}


def baseline():
    started = time.time(); common.RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = common.load_metadata(); train = common.load_split("train", metadata); val = common.load_split("validation", metadata)
    x_train, names = common.phase1_matrix(train); x_val, _ = common.phase1_matrix(val)
    cat_train, cat_val, vocab = common.encode_categories(train, val)
    frozen = common.sigmoid(val["lr_offset"])
    x_train_i, interaction_names = add_interactions(x_train, names, train)
    x_val_i, _ = add_interactions(x_val, names, val)
    lr_score, lr_config = fit_interaction_lr(train, val, x_train_i, x_val_i, cat_train, cat_val)
    results = {
        "frozen_utc_lr": {**common.metrics(val["label"], frozen),
                          "threshold": common.best_f1_threshold(val["label"], frozen)},
        "lr_explicit_interactions": {**common.metrics(val["label"], lr_score),
                                     "threshold": common.best_f1_threshold(val["label"], lr_score),
                                     "config": lr_config},
        "vocabularies": vocab, "feature_count_no_categories": len(interaction_names),
        "runtime_seconds": time.time() - started,
        "peak_rss_bytes": psutil.Process().memory_info().rss,
    }
    write_scores("frozen_utc_lr", val, frozen); write_scores("lr_explicit_interactions", val, lr_score)
    write_json(common.RESULTS / "baseline_results.json", results)
    np.save(common.RESULTS / "lr_interaction_validation_scores.npy", lr_score.astype(np.float32))
    print(json.dumps(results, sort_keys=True)); print("PHASE1_BASELINE_STATUS=SUCCEEDED")


def configurations(model, count):
    rng = random.Random(SEED + {"lightgbm": 11, "xgboost": 22, "catboost": 33}[model])
    configs = []
    for trial in range(count):
        base = {"trial": trial, "seed": SEED + trial, "max_depth": rng.randint(3, 8),
                "learning_rate": 10 ** rng.uniform(-1.7, -1.30103),
                "subsample": rng.uniform(.65, 1.0), "colsample": rng.uniform(.65, 1.0),
                "l2": 10 ** rng.uniform(-1, 2), "min_child": rng.randint(200, 800)}
        configs.append(base)
    return configs


def fit_lgb(config, x_train, y_train, x_val, y_val, cat_count):
    import lightgbm as lgb
    params = {"objective": "binary", "metric": "average_precision", "verbosity": -1,
              "num_threads": worker_threads(), "seed": config["seed"],
              "max_depth": config["max_depth"], "num_leaves": min(2 ** config["max_depth"], 127),
              "learning_rate": config["learning_rate"], "bagging_fraction": config["subsample"],
              "bagging_freq": 1, "feature_fraction": config["colsample"], "lambda_l2": config["l2"],
              "min_data_in_leaf": config["min_child"], "max_bin": 127}
    cats = list(range(x_train.shape[1] - cat_count, x_train.shape[1]))
    train_set = lgb.Dataset(x_train, y_train, categorical_feature=cats, free_raw_data=False)
    val_set = lgb.Dataset(x_val, y_val, reference=train_set, categorical_feature=cats, free_raw_data=False)
    booster = lgb.train(params, train_set, 400, valid_sets=[val_set],
                        callbacks=[lgb.early_stopping(25, verbose=False)])
    return booster, booster.predict(x_val, num_iteration=booster.best_iteration), booster.best_iteration


def fit_xgb(config, x_train, y_train, x_val, y_val, cat_count, prepared=None):
    import xgboost as xgb
    if prepared is None:
        types = ["q"] * (x_train.shape[1] - cat_count) + ["c"] * cat_count
        train_set = xgb.QuantileDMatrix(x_train, y_train, feature_types=types,
                                        enable_categorical=True, max_bin=128)
        val_set = xgb.QuantileDMatrix(x_val, y_val, ref=train_set, feature_types=types,
                                      enable_categorical=True, max_bin=128)
    else:
        train_set, val_set = prepared
    params = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
              "max_depth": config["max_depth"], "eta": config["learning_rate"],
              "subsample": config["subsample"], "colsample_bytree": config["colsample"],
              "reg_lambda": config["l2"], "min_child_weight": config["min_child"],
              "max_bin": 128, "seed": config["seed"], "nthread": worker_threads()}
    booster = xgb.train(params, train_set, 400, evals=[(val_set, "validation")],
                        early_stopping_rounds=25, verbose_eval=False)
    score = booster.predict(val_set, iteration_range=(0, booster.best_iteration + 1))
    return booster, score, booster.best_iteration + 1


def cat_object(codes):
    output = np.empty(codes.shape, dtype=object)
    for column in range(codes.shape[1]):
        vocabulary = np.asarray([str(value).encode("ascii") for value in np.unique(codes[:, column])], dtype=object)
        mapping = {int(value): token for value, token in zip(np.unique(codes[:, column]), vocabulary)}
        output[:, column] = np.fromiter((mapping[int(value)] for value in codes[:, column]), dtype=object, count=len(codes))
    return output


def fit_cat(config, numeric_train, cats_train, y_train, numeric_val, cats_val, y_val, prepared=None):
    from catboost import CatBoostClassifier, FeaturesData, Pool
    if prepared is None:
        train_data = FeaturesData(num_feature_data=numeric_train, cat_feature_data=cat_object(cats_train))
        val_data = FeaturesData(num_feature_data=numeric_val, cat_feature_data=cat_object(cats_val))
        train_pool = Pool(train_data, y_train); val_pool = Pool(val_data, y_val)
    else:
        train_pool, val_pool = prepared
    model = CatBoostClassifier(iterations=400, depth=config["max_depth"], learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2"], random_seed=config["seed"], loss_function="Logloss",
        eval_metric="PRAUC:type=Classic", bootstrap_type="Bernoulli", subsample=config["subsample"],
        rsm=config["colsample"], min_data_in_leaf=config["min_child"], grow_policy="Lossguide",
        thread_count=worker_threads(), verbose=False, allow_writing_files=False,
        od_type="Iter", od_wait=25, use_best_model=True)
    model.fit(train_pool, eval_set=val_pool)
    return model, model.predict_proba(val_pool)[:, 1], model.get_best_iteration() + 1


def search(model_name, trial_count):
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = common.load_metadata(); train = common.load_split("train", metadata); val = common.load_split("validation", metadata)
    numeric_train, names = common.phase1_matrix(train); numeric_val, _ = common.phase1_matrix(val)
    numeric_train, names = add_interactions(numeric_train, names, train)
    numeric_val, _ = add_interactions(numeric_val, common.phase1_matrix(val)[1], val)
    cat_train, cat_val, _ = common.encode_categories(train, val)
    fit = train["fit_eligible"]
    numeric_fit, cat_fit, y_fit = numeric_train[fit], cat_train[fit], train["label"][fit]
    all_train = np.column_stack([numeric_fit, cat_fit]).astype(np.float32)
    all_val = np.column_stack([numeric_val, cat_val]).astype(np.float32)
    prepared = None
    if model_name == "xgboost":
        import xgboost as xgb
        types = ["q"] * numeric_fit.shape[1] + ["c"] * cat_fit.shape[1]
        train_matrix = xgb.QuantileDMatrix(all_train, y_fit, feature_types=types,
                                           enable_categorical=True, max_bin=128)
        val_matrix = xgb.QuantileDMatrix(all_val, val["label"], ref=train_matrix,
                                         feature_types=types, enable_categorical=True, max_bin=128)
        prepared = (train_matrix, val_matrix)
    elif model_name == "catboost":
        from catboost import FeaturesData, Pool
        prepared = (Pool(FeaturesData(num_feature_data=numeric_fit, cat_feature_data=cat_object(cat_fit)), y_fit),
                    Pool(FeaturesData(num_feature_data=numeric_val, cat_feature_data=cat_object(cat_val)), val["label"]))
    log_path = common.RESULTS / f"{model_name}_search.csv"
    fieldnames = ["model", "trial", "status", "pr_auc", "roc_auc", "best_iteration", "seconds",
                  "peak_rss_bytes", "config", "error"]
    completed = set()
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as handle:
            completed = {int(row["trial"]) for row in csv.DictReader(handle) if row["status"] == "SUCCEEDED"}
    best = None
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0: writer.writeheader()
        for config in configurations(model_name, trial_count):
            if config["trial"] in completed: continue
            started = time.time(); row = {"model": model_name, "trial": config["trial"], "config": json.dumps(config)}
            try:
                if model_name == "lightgbm":
                    fitted, score, best_iteration = fit_lgb(config, all_train, y_fit, all_val, val["label"], cat_train.shape[1])
                elif model_name == "xgboost":
                    fitted, score, best_iteration = fit_xgb(config, all_train, y_fit, all_val, val["label"], cat_train.shape[1], prepared)
                else:
                    fitted, score, best_iteration = fit_cat(config, numeric_fit, cat_fit, y_fit,
                                                             numeric_val, cat_val, val["label"], prepared)
                measured = common.metrics(val["label"], score)
                row.update(status="SUCCEEDED", pr_auc=measured["pr_auc"], roc_auc=measured["roc_auc"],
                           best_iteration=best_iteration, error="")
                if best is None or measured["pr_auc"] > best[0]:
                    best = (measured["pr_auc"], config, fitted, score.astype(np.float32), best_iteration, measured)
            except Exception as exc:
                row.update(status="FAILED", pr_auc="", roc_auc="", best_iteration="",
                           error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            row.update(seconds=time.time() - started, peak_rss_bytes=psutil.Process().memory_info().rss)
            writer.writerow(row); handle.flush(); print("TRIAL=" + json.dumps(row, default=str), flush=True)
            gc.collect()
    if best is None:
        raise RuntimeError(f"no successful {model_name} trial")
    pr, config, fitted, score, iteration, measured = best
    write_scores(model_name, val, score)
    np.save(common.RESULTS / f"{model_name}_validation_scores.npy", score)
    with (common.RESULTS / f"{model_name}_best_model.pkl").open("wb") as handle: pickle.dump(fitted, handle)
    summary = {"model": model_name, "best_validation": measured, "config": config,
               "best_iteration": iteration, "threshold": common.best_f1_threshold(val["label"], score),
               "calibration": common.calibration(val["label"], score)}
    write_json(common.RESULTS / f"{model_name}_best.json", summary)
    print(json.dumps(summary, sort_keys=True)); print(f"PHASE1_{model_name.upper()}_STATUS=SUCCEEDED")


if __name__ == "__main__":
    options = cli()
    if options.stage == "baseline": baseline()
    elif not options.model: raise SystemExit("--model required for search")
    else: search(options.model, options.trials)
