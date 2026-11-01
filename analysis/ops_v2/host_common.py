"""Memory-conscious host data loading and evaluation helpers."""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("results/experiments/ml_redesign/ops_v2")
EXPORT = Path(os.environ.get("OPS_V2_EXPORT", r"C:\bda\ops_v2\pretest"))
META = ROOT / "host_export" / "pretest_metadata.json"
RESULTS = ROOT / "host_models"


def load_metadata():
    return json.loads(META.read_text(encoding="utf-8"))


def _numpy(column, dtype):
    return np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def load_split(split, metadata=None):
    metadata = metadata or load_metadata()
    path = EXPORT / f"Split={split}"
    dataset = ds.dataset(str(path), format="parquet")
    dense = list(metadata["dense_columns"])
    state = list(metadata["state_columns"])
    cats = list(metadata["categorical_columns"])
    fixed = ["FlightKey", "FlightDate", "label", "arr_minutes", "dep_del15", "arr_bucket",
             "lr_offset", "FitEligible"]
    table = dataset.to_table(columns=list(dict.fromkeys(fixed + dense + state + cats)), use_threads=True)
    n = table.num_rows
    numeric = np.empty((n, len(dense)), dtype=np.float32)
    for index, name in enumerate(dense):
        numeric[:, index] = _numpy(table[name], np.float32)
    sequences = np.empty((n, len(state)), dtype=np.float32)
    for index, name in enumerate(state):
        sequences[:, index] = _numpy(table[name], np.float32)
    raw_categories = {}
    for name in cats:
        raw_categories[name] = table[name].combine_chunks().to_pandas().astype("string").fillna("<UNK>")
    return {
        "key": table["FlightKey"].combine_chunks().to_pylist(),
        "date": _numpy(table["FlightDate"].cast("date32"), "datetime64[D]"),
        "label": _numpy(table["label"], np.int8),
        "arr_minutes": _numpy(table["arr_minutes"], np.float32),
        "dep_del15": _numpy(table["dep_del15"], np.int8),
        "arr_bucket": _numpy(table["arr_bucket"], np.int8),
        "lr_offset": _numpy(table["lr_offset"], np.float32),
        "fit_eligible": _numpy(table["FitEligible"], np.int8).astype(bool),
        "numeric": numeric, "numeric_names": dense,
        "sequence_flat": sequences, "sequence_names": state,
        "categories_raw": raw_categories,
    }


def encode_categories(train, apply):
    train_codes, apply_codes, vocabularies = [], [], {}
    for name in train["categories_raw"]:
        values = train["categories_raw"][name]
        vocabulary = sorted(set(values.tolist()))
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        train_codes.append(values.map(mapping).fillna(0).to_numpy(np.int32))
        apply_codes.append(apply["categories_raw"][name].map(mapping).fillna(0).to_numpy(np.int32))
        vocabularies[name] = {"size_including_unk": len(vocabulary) + 1}
    return np.column_stack(train_codes), np.column_stack(apply_codes), vocabularies


def sequence_summaries(data):
    names = data["sequence_names"]
    groups = {}
    for index, name in enumerate(names):
        base, hour = name.rsplit("_h", 1)
        groups.setdefault(base, {})[int(hour)] = index
    result, result_names = [], []
    for base in sorted(groups):
        indices = [groups[base][hour] for hour in sorted(groups[base])]
        values = data["sequence_flat"][:, indices]
        result.extend([values[:, 0], values.mean(1), values.max(1), values[:, 0] - values[:, 1:3].mean(1)])
        result_names.extend([base + "__recent", base + "__mean6", base + "__max6", base + "__trend1v2"])
    return np.column_stack(result).astype(np.float32), result_names


def phase1_matrix(data):
    summaries, summary_names = sequence_summaries(data)
    return np.column_stack([data["numeric"], summaries]).astype(np.float32), data["numeric_names"] + summary_names


def sigmoid(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def metrics(y, score, fractions=(0.05, 0.10, 0.20, 0.30)):
    order = np.argsort(-score, kind="stable")
    prevalence = float(y.mean())
    ranking = []
    for fraction in fractions:
        n = max(1, int(round(len(y) * fraction)))
        selected = y[order[:n]]
        precision = float(selected.mean())
        ranking.append({"fraction": fraction, "precision": precision,
                        "recall": float(selected.sum() / max(y.sum(), 1)),
                        "lift": precision / prevalence})
    return {"pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "prevalence": prevalence,
            "ranking": ranking}


def calibration(y, score, bins=10):
    order = np.argsort(score, kind="stable")
    rows = []
    for index, ids in enumerate(np.array_split(order, bins), 1):
        rows.append({"bin": index, "rows": len(ids), "mean_score": float(score[ids].mean()),
                     "observed_rate": float(y[ids].mean())})
    return rows


def best_f1_threshold(y, score):
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = score >= threshold
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[0]: best = (f1, float(threshold), precision, recall)
    return {"f1": best[0], "threshold": best[1], "precision": best[2], "recall": best[3]}


def date_block_bootstrap(dates, y, candidate, reference, replicates=1000, seed=20261101):
    unique = np.unique(dates)
    blocks = {date: np.where(dates == date)[0] for date in unique}
    rng = np.random.default_rng(seed); differences = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([blocks[date] for date in sampled])
        differences.append(average_precision_score(y[ids], candidate[ids]) -
                           average_precision_score(y[ids], reference[ids]))
    values = np.asarray(differences)
    return {"replicates": replicates, "seed": seed, "date_blocks": len(unique),
            "mean": float(values.mean()), "ci_95_low": float(np.quantile(values, .025)),
            "ci_95_high": float(np.quantile(values, .975))}
