#!/usr/bin/env python3
"""Post-freeze FOURTH DESIGN EVALUATION.  Never invoke before the gate commit."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.nn import functional as F

import dl_host
import host_common as common
import phase1_host as phase1
from dl_model import FlightDelayResidualNet

FROZEN = Path("results/experiments/ml_redesign/ops_v2/frozen_ops_v2_config.json")
OUT = Path("results/experiments/ml_redesign/ops_v2/fourth_design")


def combine(first, second):
    n1, n2 = len(first["label"]), len(second["label"]); result = {}
    for key, value in first.items():
        other = second[key]
        if key == "categories_raw":
            result[key] = {name: pd.concat([value[name], other[name]], ignore_index=True)
                           for name in value}
        elif isinstance(value, np.ndarray) and len(value) == n1 and isinstance(other, np.ndarray) and len(other) == n2:
            result[key] = np.concatenate([value, other])
        elif isinstance(value, list) and len(value) == n1 and isinstance(other, list) and len(other) == n2:
            result[key] = value + other
        else:
            result[key] = value
    return result


def phase1_fixed(name, fit, test, config, iterations):
    x_fit, names = common.phase1_matrix(fit); x_test, _ = common.phase1_matrix(test)
    x_fit, names = phase1.add_interactions(x_fit, names, fit)
    x_test, _ = phase1.add_interactions(x_test, common.phase1_matrix(test)[1], test)
    cat_fit, cat_test, _ = common.encode_categories(fit, test)
    eligible = fit["fit_eligible"]
    numeric_fit, cat_fit_e, y = x_fit[eligible], cat_fit[eligible], fit["label"][eligible]
    if name == "lightgbm":
        import lightgbm as lgb
        all_fit = np.column_stack([numeric_fit, cat_fit_e]).astype(np.float32)
        all_test = np.column_stack([x_test, cat_test]).astype(np.float32)
        cats = list(range(numeric_fit.shape[1], all_fit.shape[1]))
        params = {"objective": "binary", "metric": "average_precision", "verbosity": -1,
                  "num_threads": 11, "seed": config["seed"], "max_depth": config["max_depth"],
                  "num_leaves": min(2 ** config["max_depth"], 127), "learning_rate": config["learning_rate"],
                  "bagging_fraction": config["subsample"], "bagging_freq": 1,
                  "feature_fraction": config["colsample"], "lambda_l2": config["l2"],
                  "min_data_in_leaf": config["min_child"], "max_bin": 127}
        model = lgb.train(params, lgb.Dataset(all_fit, y, categorical_feature=cats), num_boost_round=iterations)
        return model.predict(all_test)
    if name == "xgboost":
        import xgboost as xgb
        all_fit = np.column_stack([numeric_fit, cat_fit_e]).astype(np.float32)
        all_test = np.column_stack([x_test, cat_test]).astype(np.float32)
        types = ["q"] * numeric_fit.shape[1] + ["c"] * cat_fit_e.shape[1]
        params = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
                  "max_depth": config["max_depth"], "eta": config["learning_rate"],
                  "subsample": config["subsample"], "colsample_bytree": config["colsample"],
                  "reg_lambda": config["l2"], "min_child_weight": config["min_child"],
                  "max_bin": 128, "seed": config["seed"], "nthread": 11}
        train_matrix = xgb.QuantileDMatrix(all_fit, y, feature_types=types, enable_categorical=True)
        test_matrix = xgb.QuantileDMatrix(all_test, ref=train_matrix, feature_types=types, enable_categorical=True)
        return xgb.train(params, train_matrix, num_boost_round=iterations).predict(test_matrix)
    from catboost import CatBoostClassifier, FeaturesData, Pool
    train_pool = Pool(FeaturesData(num_feature_data=numeric_fit,
                                   cat_feature_data=phase1.cat_object(cat_fit_e)), y)
    test_pool = Pool(FeaturesData(num_feature_data=x_test,
                                  cat_feature_data=phase1.cat_object(cat_test)))
    model = CatBoostClassifier(iterations=iterations, depth=config["max_depth"],
        learning_rate=config["learning_rate"], l2_leaf_reg=config["l2"], random_seed=config["seed"],
        loss_function="Logloss", bootstrap_type="Bernoulli", subsample=config["subsample"],
        rsm=config["colsample"], min_data_in_leaf=config["min_child"], grow_policy="Lossguide",
        thread_count=11, verbose=False, allow_writing_files=False)
    model.fit(train_pool)
    return model.predict_proba(test_pool)[:, 1]


def prepare_dl(fit, test):
    eligible = fit["fit_eligible"]
    mean = fit["numeric"][eligible].mean(0, dtype=np.float64).astype(np.float32)
    std = fit["numeric"][eligible].std(0, dtype=np.float64).astype(np.float32); std[std < 1e-6] = 1
    fit["dense_scaled"] = ((fit["numeric"] - mean) / std).astype(np.float32)
    test["dense_scaled"] = ((test["numeric"] - mean) / std).astype(np.float32)
    fit["categories"], test["categories"], vocab = dl_host.category_arrays(fit, test)
    fit["states"] = dl_host.state_arrays(fit); test["states"] = dl_host.state_arrays(test)
    return vocab


def dl_fixed_once(fit, test, vocab, config, seed, epochs):
    dl_host.set_seed(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_channels = {name: values.shape[2] for name, values in fit["states"].items()}
    model = FlightDelayResidualNet(fit["dense_scaled"].shape[1], vocab, state_channels,
        width=config["width"], dropout=config["dropout"], embedding_scale=config["embedding_scale"],
        state_kind=config["state_kind"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    ids = np.flatnonzero(fit["fit_eligible"]); batch_size = 4096
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"], total_steps=math.ceil(len(ids) / batch_size) * epochs)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train(); rng.shuffle(ids)
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            output = model(*dl_host.tensors(fit, batch, device), apply_id_dropout=True)
            y = torch.from_numpy(fit["label"][batch].astype(np.float32)).to(device)
            minutes = torch.from_numpy(fit["arr_minutes"][batch] / 60.0).to(device)
            dep = torch.from_numpy(fit["dep_del15"][batch].astype(np.float32)).to(device)
            bucket = torch.from_numpy(fit["arr_bucket"][batch].astype(np.int64)).to(device)
            scale = config["aux_scale"]
            loss = F.binary_cross_entropy_with_logits(output["main"], y) + scale * (
                .3 * F.huber_loss(output["minutes"], minutes, delta=1.0)
                + .2 * F.binary_cross_entropy_with_logits(output["dep"], dep)
                + .2 * F.cross_entropy(output["bucket"], bucket))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
    return dl_host.predict(model, test, device)


def sliced_metrics(test, score):
    rows = {"all_test": common.metrics(test["label"], score)}
    holiday = test["categories_raw"]["HolidayPeriod"].str.contains("Christmas", case=False).to_numpy()
    for name, mask in [("christmas", holiday), ("regular", ~holiday)]:
        if mask.any() and np.unique(test["label"][mask]).size > 1:
            rows[name] = common.metrics(test["label"][mask], score[mask])
    return rows


def main():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if not frozen.get("decisions_frozen") or not frozen.get("committed_before_test"):
        raise RuntimeError("frozen config is absent or not marked committed; refusing to open test")
    if not (frozen["phase1_passed"] or frozen["phase2_passed"]):
        raise RuntimeError("neither decision gate passed; test access is forbidden")
    print("DECISIONS_FROZEN=1", flush=True)
    metadata = common.load_metadata(); train = common.load_split("train", metadata)
    validation = common.load_split("validation", metadata); test = common.load_split("test", metadata)
    fit = combine(train, validation); OUT.mkdir(parents=True, exist_ok=True)
    phase1_best = json.loads((common.RESULTS / f"{frozen['phase1_model']}_best.json").read_text(encoding="utf-8"))
    phase1_score = phase1_fixed(frozen["phase1_model"], fit, test, phase1_best["config"],
                                int(phase1_best["best_iteration"]))
    scores = {"utc_lr_refit": common.sigmoid(test["lr_offset"]), frozen["phase1_model"]: phase1_score}
    if frozen.get("run_dl_test"):
        config = dl_host.read_best(); vocab = prepare_dl(fit, test); epochs = int(config["best_epoch"])
        seed_scores = [dl_fixed_once(fit, test, vocab, config, seed, epochs) for seed in dl_host.SEEDS]
        logits = [np.log(np.clip(score, 1e-7, 1-1e-7) / np.clip(1-score, 1e-7, 1)) for score in seed_scores]
        scores["deep_ensemble"] = common.sigmoid(np.mean(np.stack(logits), axis=0))
    results = {name: sliced_metrics(test, score) for name, score in scores.items()}
    reference = scores["utc_lr_refit"]
    results["bootstrap_vs_utc_lr"] = {name: common.date_block_bootstrap(
        test["date"], test["label"], score, reference, 1000, 20261101)
        for name, score in scores.items() if name != "utc_lr_refit"}
    results["label"] = "FOURTH DESIGN EVALUATION - test used for the fourth time; no decision used it"
    for name, score in scores.items(): np.save(OUT / f"{name}_test_scores.npy", score.astype(np.float32))
    (OUT / "test_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, sort_keys=True)); print("FOURTH_DESIGN_EVALUATION_STATUS=SUCCEEDED")


if __name__ == "__main__":
    main()
