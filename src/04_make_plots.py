#!/usr/bin/env python3
"""Create report-ready PNG plots from the ML pipeline's small CSV outputs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot flight-delay ML result tables")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def require(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required result table not found: {path}")
    return pd.read_csv(path)


def main():
    args = parse_args()
    source = args.results_dir.resolve()
    output = (args.output_dir or (source / "plots")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics = require(source / "metrics.csv")
    metric_names = ["AUC_ROC", "AUC_PR", "F1_class1", "Precision_class1", "Recall_class1", "Accuracy"]
    plot_data = metrics.set_index("Model")[metric_names].apply(pd.to_numeric)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_data.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title("Held-out test performance by model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    save(fig, output / "model_metrics.png")

    baseline_path = source / "baseline_metrics.csv"
    if baseline_path.is_file():
        baselines = pd.read_csv(baseline_path)
        comparison = pd.concat([metrics, baselines], ignore_index=True)
        selected = ["AUC_PR", "F1_class1", "Precision_class1", "Recall_class1", "Accuracy"]
        baseline_data = comparison.set_index("Model")[selected].apply(pd.to_numeric)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        baseline_data.plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Metric value")
        ax.set_title("Held-out models versus simple baselines")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        save(fig, output / "model_vs_baselines.png")

    confusion = require(source / "confusion_counts.csv")
    models = list(confusion["Model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        part = confusion[confusion.Model == model].copy()
        matrix = part.pivot(index="ActualLabel", columns="PredictedLabel", values="Count").fillna(0)
        matrix = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0).astype(int)
        image = ax.imshow(matrix, cmap="Blues")
        for (row, column), value in pd.DataFrame(matrix).stack().items():
            ax.text(column, row, f"{int(value):,}", ha="center", va="center")
        ax.set_title(model)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, output / "confusion_matrices.png")

    experiments_path = source / "experiment_comparison.csv"
    if experiments_path.is_file():
        experiments = pd.read_csv(experiments_path)
        if "Role" in experiments.columns:
            experiments = experiments[experiments["Role"] == "feature experiment"]
        experiments["ValidationAUC_PR"] = pd.to_numeric(experiments["ValidationAUC_PR"])
        experiments = experiments.sort_values("ValidationAUC_PR")
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(experiments))))
        ax.barh(experiments["Experiment"], experiments["ValidationAUC_PR"], color="#4472c4")
        ax.set_xlabel("Validation AUC-PR")
        ax.set_title("Controlled feature experiments")
        ax.grid(axis="x", alpha=0.25)
        save(fig, output / "experiment_validation_auc_pr.png")

    importance_path = source / "random_forest_feature_importance_top30.csv"
    if importance_path.is_file():
        importance = pd.read_csv(importance_path).head(20).copy()
        importance["Importance"] = pd.to_numeric(importance["Importance"])
        importance = importance.sort_values("Importance")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(importance["Feature"], importance["Importance"], color="#70ad47")
        ax.set_xlabel("Random Forest importance")
        ax.set_title("Top 20 Random Forest feature importances")
        ax.grid(axis="x", alpha=0.25)
        save(fig, output / "random_forest_feature_importance.png")

    counts_path = source / "class_counts.csv"
    if counts_path.is_file():
        counts = pd.read_csv(counts_path)
        counts["Flights"] = pd.to_numeric(counts["Flights"])
        counts["Delayed15Plus"] = pd.to_numeric(counts["Delayed15Plus"])
        counts["PositiveRate"] = counts["Delayed15Plus"] / counts["Flights"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(counts["Split"], counts["PositiveRate"], color=["#4472c4", "#ed7d31", "#a5a5a5"])
        ax.set_ylim(0, max(0.25, counts["PositiveRate"].max() * 1.2))
        ax.set_ylabel("Delayed-flight prevalence")
        ax.set_title("Positive-class prevalence by chronological split")
        for bar, value in zip(bars, counts["PositiveRate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1%}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.25)
        save(fig, output / "prevalence_by_split.png")

    print("ML_PLOTS_STATUS=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
