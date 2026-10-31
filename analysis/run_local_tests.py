#!/usr/bin/env python3
"""Run both tracked local-Spark tests and preserve exact logs and timings."""

import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "ml_redesign" / "diagnostics"
TESTS = [
    ("temporal_history_features", "/src/tests/test_temporal_history_features.py", "TEMPORAL_HISTORY_TEST_STATUS=PASSED"),
    ("ml_pipeline", "/src/tests/test_ml_pipeline.py", "ML_PIPELINE_TESTS_STATUS=PASSED"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summarize_existing = "--summarize-existing" in sys.argv
    previous = {}
    summary_path = OUT / "local_test_runs.csv"
    if summarize_existing and summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            previous = {row["Test"]: row for row in csv.DictReader(handle)}
    rows = []
    for name, script, expected in TESTS:
        command = (f'spark-submit --master "local[2]" --driver-memory 2g '
                   f'--conf spark.eventLog.enabled=false --conf spark.ui.enabled=false {script}')
        full = ["docker", "exec", "master", "bash", "-lc", command]
        log_path = OUT / f"local_test_{name}.log"
        if summarize_existing:
            if not log_path.exists():
                raise SystemExit(f"Missing existing log: {log_path}")
            old = previous.get(name, {})
            started = old.get("StartUTC", "recorded in log")
            elapsed = old.get("WallTimeMs", "recorded in log")
            exit_code = int(old.get("ExitCode", 0))
        else:
            started = datetime.now(timezone.utc).isoformat()
            before = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(full, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            elapsed = round((time.monotonic() - before) * 1000)
            exit_code = process.returncode
        content = log_path.read_text(encoding="utf-8", errors="replace")
        marker_seen = expected in content
        result = "PASSED" if exit_code == 0 and marker_seen else "FAILED"
        rows.append([name, started, elapsed, exit_code, expected, int(marker_seen), result, command, str(log_path.relative_to(ROOT))])
        print(f"{name}: {result} exit={exit_code} elapsed_ms={elapsed}", flush=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Test", "StartUTC", "WallTimeMs", "ExitCode", "ExpectedMarker", "MarkerSeen", "Result", "Command", "Log"])
        writer.writerows(rows)
    if any(row[6] != "PASSED" for row in rows):
        raise SystemExit(1)
    print("ALL_LOCAL_TESTS_STATUS=PASSED")


if __name__ == "__main__":
    main()
