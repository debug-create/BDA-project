#!/usr/bin/env python3
"""Five-seed rolling-origin evaluation of the selected deep architecture."""

import json

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import dl_host
import host_common as common
import phase1_host as phase1
from phase1_evaluate import subset


def fold_prepare(fit_data, holdout):
    fit = fit_data["fit_eligible"]
    mean = fit_data["numeric"][fit].mean(0, dtype=np.float64).astype(np.float32)
    std = fit_data["numeric"][fit].std(0, dtype=np.float64).astype(np.float32); std[std < 1e-6] = 1
    fit_data["dense_scaled"] = ((fit_data["numeric"] - mean) / std).astype(np.float32)
    holdout["dense_scaled"] = ((holdout["numeric"] - mean) / std).astype(np.float32)
    fit_data["categories"], holdout["categories"], vocab = dl_host.category_arrays(fit_data, holdout)
    fit_data["states"] = dl_host.state_arrays(fit_data); holdout["states"] = dl_host.state_arrays(holdout)

    # Fold-specific LR offset: all transforms and coefficients use earlier rows only.
    x_fit, _ = common.phase1_matrix(fit_data); x_hold, _ = common.phase1_matrix(holdout)
    cat_fit, cat_hold, _ = common.encode_categories(fit_data, holdout)
    scaler = StandardScaler().fit(x_fit[fit])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32).fit(cat_fit[fit])
    a = sparse.hstack([sparse.csr_matrix(scaler.transform(x_fit[fit]).astype(np.float32)),
                       encoder.transform(cat_fit[fit])], format="csr")
    all_fit = sparse.hstack([sparse.csr_matrix(scaler.transform(x_fit).astype(np.float32)),
                             encoder.transform(cat_fit)], format="csr")
    b = sparse.hstack([sparse.csr_matrix(scaler.transform(x_hold).astype(np.float32)),
                       encoder.transform(cat_hold)], format="csr")
    lr = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, max_iter=40, tol=1e-4,
                       average=True, random_state=20261101, n_jobs=-1).fit(a, fit_data["label"][fit])
    fit_p = np.clip(lr.predict_proba(all_fit)[:, 1], 1e-7, 1 - 1e-7)
    hold_p = np.clip(lr.predict_proba(b)[:, 1], 1e-7, 1 - 1e-7)
    fit_data["lr_offset"] = np.log(fit_p / (1 - fit_p)).astype(np.float32)
    holdout["lr_offset"] = np.log(hold_p / (1 - hold_p)).astype(np.float32)
    return vocab


def main():
    metadata = common.load_metadata(); train = common.load_split("train", metadata)
    config = dl_host.read_best(); max_epochs = int(config["best_epoch"])
    rows = []
    for start_text in ["2024-10-28", "2024-11-04", "2024-11-11", "2024-11-18"]:
        start = np.datetime64(start_text); stop = start + np.timedelta64(7, "D")
        fit_ids = np.flatnonzero(train["date"] < start)
        hold_ids = np.flatnonzero((train["date"] >= start) & (train["date"] < stop))
        fit_data = subset(train, fit_ids); holdout = subset(train, hold_ids)
        vocab = fold_prepare(fit_data, holdout)
        logits, seed_rows = [], []
        for seed in dl_host.SEEDS:
            run = dl_host.train_once(fit_data, holdout, vocab, config, seed, max_epochs)
            score = run.pop("score"); run.pop("state")
            logits.append(np.log(np.clip(score, 1e-7, 1 - 1e-7) / np.clip(1 - score, 1e-7, 1)))
            seed_rows.append({"seed": seed, **run})
        deep_score = common.sigmoid(np.mean(np.stack(logits), axis=0))
        phase1_scores = np.load(common.RESULTS / f"fold_{start_text}.npz")
        candidate = phase1_scores["candidate_score"]
        deep_metrics = common.metrics(holdout["label"], deep_score)
        candidate_metrics = common.metrics(holdout["label"], candidate)
        row = {"validation_start": start_text, "validation_end": str(stop - np.timedelta64(1, "D")),
               "fit_rows": len(fit_ids), "validation_rows": len(hold_ids), "seeds": seed_rows,
               "phase1_pr_auc": candidate_metrics["pr_auc"], "dl_pr_auc": deep_metrics["pr_auc"],
               "pr_auc_gain": deep_metrics["pr_auc"] - candidate_metrics["pr_auc"],
               "phase1_top10_precision": candidate_metrics["ranking"][1]["precision"],
               "dl_top10_precision": deep_metrics["ranking"][1]["precision"]}
        rows.append(row)
        (dl_host.RESULTS / "rolling_folds.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print("DL_FOLD=" + json.dumps(row), flush=True)
    gains = np.asarray([row["pr_auc_gain"] for row in rows])
    summary = {"folds": rows, "mean_pr_auc_gain": float(gains.mean()),
               "std_pr_auc_gain": float(gains.std(ddof=1))}
    (dl_host.RESULTS / "rolling_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PHASE2_FOLDS_STATUS=SUCCEEDED")


if __name__ == "__main__":
    main()
