#!/usr/bin/env python3
"""Apply the Phase-2 validation/bootstrap/fold decision rule without test access."""

import json

import numpy as np

import dl_host
import host_common as common


def main():
    metadata = common.load_metadata(); validation = common.load_split("validation", metadata)
    phase1 = json.loads((common.RESULTS / "phase1_decision.json").read_text(encoding="utf-8"))
    phase1_score = np.load(common.RESULTS / f"{phase1['winner']}_validation_scores.npy")
    dl_score = np.load(dl_host.RESULTS / "full_validation_scores.npy")
    full = json.loads((dl_host.RESULTS / "full_summary.json").read_text(encoding="utf-8"))
    ablations = json.loads((dl_host.RESULTS / "ablation_summary.json").read_text(encoding="utf-8"))
    rolling = json.loads((dl_host.RESULTS / "rolling_summary.json").read_text(encoding="utf-8"))
    bootstrap = common.date_block_bootstrap(validation["date"], validation["label"],
                                             dl_score, phase1_score, replicates=1000, seed=20261101)
    full_pr = full["ensemble"]["pr_auc"]
    drops = {name: full_pr - result["ensemble"]["pr_auc"] for name, result in ablations.items()}
    carrying_component = max(drops, key=drops.get)
    ablation_evidence = drops[carrying_component] > 0
    passed = bool(rolling["mean_pr_auc_gain"] >= .003 and bootstrap["ci_95_low"] > 0 and ablation_evidence)
    result = {
        "phase1_reference": phase1["winner"],
        "validation_pr_auc_phase1": float(common.metrics(validation["label"], phase1_score)["pr_auc"]),
        "validation_pr_auc_dl": float(common.metrics(validation["label"], dl_score)["pr_auc"]),
        "bootstrap_dl_minus_phase1": bootstrap,
        "rolling_mean_pr_auc_gain": rolling["mean_pr_auc_gain"],
        "rolling_std_pr_auc_gain": rolling["std_pr_auc_gain"],
        "ablation_pr_auc_drops": drops,
        "carrying_component": carrying_component,
        "ablation_evidence": ablation_evidence,
        "decision_rule_passed": passed,
        "test_authorized_by_phase2": passed,
        "test_opened": False,
    }
    (dl_host.RESULTS / "phase2_decision.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); print("PHASE2_EVALUATION_STATUS=SUCCEEDED")


if __name__ == "__main__":
    main()
