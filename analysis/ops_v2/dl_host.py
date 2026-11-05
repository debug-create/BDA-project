#!/usr/bin/env python3
"""Validation-only Phase-2 search and ablations on the sealed pretest export."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

import host_common as common
from dl_model import EMBED_DIMS, FlightDelayResidualNet, parameter_count

SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
RESULTS = Path("results/experiments/ml_redesign/ops_v2/dl")
CATEGORY_NAMES = list(EMBED_DIMS)
STREAMS = ["origin_departures", "origin_inbound", "carrier_network", "destination_arrivals"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["search", "ensemble", "ablations", "summarize-ablations"], required=True)
    parser.add_argument("--configs", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--variant", default="full",
                        choices=["full", "no_aux", "no_states", "no_offset",
                                 "no_embeddings", "gru", "plain_mlp"])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def category_arrays(train, validation):
    encoded_train, encoded_validation, all_vocab = common.encode_categories(train, validation)
    all_names = list(train["categories_raw"])
    indices = [all_names.index(name) for name in CATEGORY_NAMES]
    train_selected = encoded_train[:, indices]
    validation_selected = encoded_validation[:, indices]
    hash_column = CATEGORY_NAMES.index("RouteHash65536")
    # The route hash is a fixed 2^16 mapping, not a learned vocabulary.  Zero
    # remains reserved for UNK/dropout, hence the +1 shift.
    train_selected[:, hash_column] = train["categories_raw"]["RouteHash65536"].astype(int).to_numpy(np.int32) + 1
    validation_selected[:, hash_column] = validation["categories_raw"]["RouteHash65536"].astype(int).to_numpy(np.int32) + 1
    vocab = {name: all_vocab[name]["size_including_unk"] for name in CATEGORY_NAMES}
    vocab["RouteHash65536"] = 65537
    return train_selected, validation_selected, vocab


def state_arrays(data):
    names = data["sequence_names"]
    streams = {}
    for stream in STREAMS:
        available = {name.rsplit("_h", 1)[0] for name in names if name.startswith("seq_" + stream + "_")}
        masks = sorted(name for name in available if name.endswith("_mask"))
        channel_names = sorted(available - set(masks)) + masks
        channels = []
        for channel in channel_names:
            columns = [names.index(f"{channel}_h{hour}") for hour in range(1, 7)]
            channels.append(data["sequence_flat"][:, columns])
        streams[stream] = np.stack(channels, axis=2).astype(np.float32)
    return streams


def prepare():
    metadata = common.load_metadata()
    train = common.load_split("train", metadata); validation = common.load_split("validation", metadata)
    fit = train["fit_eligible"]
    mean = train["numeric"][fit].mean(0, dtype=np.float64).astype(np.float32)
    std = train["numeric"][fit].std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1
    train["dense_scaled"] = ((train["numeric"] - mean) / std).astype(np.float32)
    validation["dense_scaled"] = ((validation["numeric"] - mean) / std).astype(np.float32)
    train["categories"], validation["categories"], vocab = category_arrays(train, validation)
    train["states"] = state_arrays(train); validation["states"] = state_arrays(validation)
    return train, validation, vocab, {"mean": mean, "std": std}


def configs(count):
    rng = random.Random(SEEDS[0] + 99)
    choices = []
    for index in range(count):
        choices.append({
            "config": index,
            "embedding_scale": rng.choice([0.75, 1.0, 1.25]),
            "dropout": rng.uniform(0.10, 0.40),
            "width": rng.choice([128, 256]),
            "lr": 10 ** rng.uniform(math.log10(3e-4), math.log10(3e-3)),
            "aux_scale": rng.choice([0.5, 1.0, 2.0]),
            "state_kind": rng.choice(["conv", "gru"]),
        })
    return choices


def tensors(data, ids, device):
    states = {name: torch.from_numpy(values[ids]).to(device) for name, values in data["states"].items()}
    return (
        torch.from_numpy(data["dense_scaled"][ids]).to(device),
        torch.from_numpy(data["categories"][ids].astype(np.int64, copy=False)).to(device),
        states,
        torch.from_numpy(data["lr_offset"][ids]).to(device),
    )


@torch.inference_mode()
def predict(model, data, device, batch_size=8192):
    model.eval(); scores = np.empty(len(data["label"]), dtype=np.float32)
    for start in range(0, len(scores), batch_size):
        ids = np.arange(start, min(start + batch_size, len(scores)))
        scores[ids] = torch.sigmoid(model(*tensors(data, ids, device))["main"]).cpu().numpy()
    return scores


def train_once(train, validation, vocab, config, seed, max_epochs, variant="full"):
    set_seed(seed); torch.set_num_threads(int(os.environ.get(
        "OPS_V2_THREADS", max(1, (psutil.cpu_count(logical=True) or 2) - 1))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_channels = {name: values.shape[2] for name, values in train["states"].items()}
    model = FlightDelayResidualNet(
        train["dense_scaled"].shape[1], vocab, state_channels,
        width=config["width"], dropout=config["dropout"], embedding_scale=config["embedding_scale"],
        state_kind="gru" if variant == "gru" else config["state_kind"],
        use_states=variant not in {"no_states", "plain_mlp"},
        use_embeddings=variant not in {"no_embeddings", "plain_mlp"},
        use_offset=variant != "no_offset",
        simple_onehot=variant == "no_embeddings",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    fit_ids = np.flatnonzero(train["fit_eligible"])
    batch_size = 4096
    steps = math.ceil(len(fit_ids) / batch_size) * max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], total_steps=steps)
    rng = np.random.default_rng(seed); best = None; patience = 0; curve = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train(); rng.shuffle(fit_ids); running = 0.0
        for start in range(0, len(fit_ids), batch_size):
            ids = fit_ids[start:start + batch_size]
            dense, categories, states, offset = tensors(train, ids, device)
            output = model(dense, categories, states, offset, apply_id_dropout=True)
            y = torch.from_numpy(train["label"][ids].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(output["main"], y)
            if variant != "no_aux":
                scale = config["aux_scale"]
                minutes = torch.from_numpy(train["arr_minutes"][ids] / 60.0).to(device)
                dep = torch.from_numpy(train["dep_del15"][ids].astype(np.float32)).to(device)
                bucket = torch.from_numpy(train["arr_bucket"][ids].astype(np.int64)).to(device)
                loss = loss + scale * (0.3 * F.huber_loss(output["minutes"], minutes, delta=1.0)
                                       + 0.2 * F.binary_cross_entropy_with_logits(output["dep"], dep)
                                       + 0.2 * F.cross_entropy(output["bucket"], bucket))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); running += float(loss.detach()) * len(ids)
        validation_score = predict(model, validation, device)
        val_pr = float(average_precision_score(validation["label"], validation_score))
        # A deterministic 200k fitting sample provides overfit evidence without another full pass.
        sample_ids = fit_ids[:min(200000, len(fit_ids))]
        sampled = {key: value for key, value in train.items()}
        sampled["label"] = train["label"][sample_ids]
        sampled["dense_scaled"] = train["dense_scaled"][sample_ids]
        sampled["categories"] = train["categories"][sample_ids]
        sampled["lr_offset"] = train["lr_offset"][sample_ids]
        sampled["states"] = {key: value[sample_ids] for key, value in train["states"].items()}
        train_score = predict(model, sampled, device)
        train_pr = float(average_precision_score(sampled["label"], train_score))
        curve.append({"epoch": epoch, "loss": running / len(fit_ids), "train_pr_auc_sample": train_pr,
                      "validation_pr_auc": val_pr})
        print(json.dumps({"epoch": epoch, "validation_pr_auc": val_pr, "seed": seed,
                          "variant": variant}), flush=True)
        if best is None or val_pr > best[0] + 1e-12:
            best = (val_pr, epoch, validation_score.copy(), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= 3:
            break
    model.load_state_dict(best[3])
    return {
        "pr_auc": best[0], "best_epoch": best[1], "score": best[2], "state": best[3],
        "curve": curve, "seconds": time.time() - started, "parameters": parameter_count(model),
        "peak_rss_bytes": psutil.Process().memory_info().rss,
    }


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_best():
    return json.loads((RESULTS / "search_best.json").read_text(encoding="utf-8"))


def search(count, max_epochs):
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, validation, vocab, scaler = prepare()
    rows = []
    for config in configs(count):
        try:
            run = train_once(train, validation, vocab, config, SEEDS[0], max_epochs)
            row = {**config, "status": "SUCCEEDED", "validation_pr_auc": run["pr_auc"],
                   "best_epoch": run["best_epoch"], "seconds": run["seconds"],
                   "peak_rss_bytes": run["peak_rss_bytes"], "error": ""}
        except Exception as exc:
            row = {**config, "status": "FAILED", "validation_pr_auc": None,
                   "best_epoch": None, "seconds": None, "peak_rss_bytes": psutil.Process().memory_info().rss,
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row); save_json(RESULTS / "search_log.json", rows); print(json.dumps(row), flush=True); gc.collect()
    successful = [row for row in rows if row["status"] == "SUCCEEDED"]
    if not successful: raise RuntimeError("all deep search configurations failed")
    best_score = max(row["validation_pr_auc"] for row in successful)
    within_margin = [row for row in successful if row["validation_pr_auc"] >= best_score - .002]
    # Apply the specified +0.002 margin in favour of the smaller architecture.
    best = min(within_margin, key=lambda row: (
        row["width"], row["embedding_scale"], row["state_kind"] == "gru", row["dropout"], row["config"]))
    best["absolute_best_validation_pr_auc"] = best_score
    best["selected_with_simplicity_margin"] = float(best["validation_pr_auc"]) < float(best_score)
    save_json(RESULTS / "search_best.json", best)
    print("PHASE2_SEARCH_STATUS=SUCCEEDED")


def ensemble(max_epochs, variant="full"):
    train, validation, vocab, scaler = prepare(); config = read_best()
    runs, scores = [], []
    epochs = min(max_epochs, int(config["best_epoch"]))
    for seed in SEEDS:
        run = train_once(train, validation, vocab, config, seed, epochs, variant=variant)
        scores.append(run.pop("score")); run.pop("state")
        runs.append({"seed": seed, **run})
        save_json(RESULTS / f"{variant}_runs.json", runs)
    ensemble_score = common.sigmoid(np.mean(np.stack([np.log(np.clip(score, 1e-7, 1-1e-7) /
                                                        np.clip(1-score, 1e-7, 1)) for score in scores]), axis=0))
    np.save(RESULTS / f"{variant}_validation_scores.npy", ensemble_score.astype(np.float32))
    seed_pr = [row["pr_auc"] for row in runs]
    summary = {"variant": variant, "seeds": SEEDS, "runs": runs,
               "seed_pr_auc_mean": float(np.mean(seed_pr)),
               "seed_pr_auc_std": float(np.std(seed_pr, ddof=1)),
               "ensemble": common.metrics(validation["label"], ensemble_score),
               "threshold": common.best_f1_threshold(validation["label"], ensemble_score),
               "calibration": common.calibration(validation["label"], ensemble_score)}
    save_json(RESULTS / f"{variant}_summary.json", summary)
    return summary


def ablations(max_epochs):
    summaries = {}
    for variant in ["no_aux", "no_states", "no_offset", "no_embeddings", "gru", "plain_mlp"]:
        summaries[variant] = ensemble(max_epochs, variant)
    save_json(RESULTS / "ablation_summary.json", summaries)
    print("PHASE2_ABLATIONS_STATUS=SUCCEEDED")


def summarize_ablations():
    variants = ["no_aux", "no_states", "no_offset", "no_embeddings", "gru", "plain_mlp"]
    summaries = {
        variant: json.loads((RESULTS / f"{variant}_summary.json").read_text(encoding="utf-8"))
        for variant in variants
    }
    save_json(RESULTS / "ablation_summary.json", summaries)
    print("PHASE2_ABLATIONS_STATUS=SUCCEEDED")


if __name__ == "__main__":
    options = arguments()
    if options.stage == "search": search(options.configs, options.max_epochs)
    elif options.stage == "ensemble":
        ensemble(options.max_epochs, options.variant); print("PHASE2_ENSEMBLE_STATUS=SUCCEEDED")
    elif options.stage == "ablations":
        ablations(options.max_epochs)
    else:
        summarize_ablations()
