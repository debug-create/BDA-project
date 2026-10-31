#!/usr/bin/env python3
"""Read-only availability gate for the earlier-flights feasibility analysis."""

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "ml_redesign" / "diagnostics"
LOCAL = Path(r"C:\bda\data")
NAMES = [f"flights_2024_{month}.csv" for month in ("10", "11", "12")]
HDFS = "/user/root/flights/raw"


def hdfs_exists(path):
    result = subprocess.run(
        ["docker", "exec", "master", "bash", "-lc", f"hdfs dfs -test -e '{path}'"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


OUT.mkdir(parents=True, exist_ok=True)
rows = []
for name in NAMES:
    local_path = LOCAL / name
    hdfs_path = f"{HDFS}/{name}"
    rows.append((name, str(local_path), int(local_path.is_file()), hdfs_path, int(hdfs_exists(hdfs_path))))

with (OUT / "part_c_raw_availability.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["RequiredFile", "ExpectedLocalPath", "ExistsLocally", "ExpectedHDFSPath", "ExistsInHDFS"])
    writer.writerows(rows)

available = all(row[2] or row[4] for row in rows)
with (OUT / "part_c_status.txt").open("w", encoding="utf-8") as handle:
    if available:
        handle.write("PART_C_GATE=AVAILABLE\n")
    else:
        handle.write("PART_C_GATE=STOPPED\n")
        handle.write("Reason: one or more required raw BTS CSVs are absent from both allowed locations.\n")
        handle.write(r"Expected local directory: C:\bda\data" + "\n")
        handle.write("Expected HDFS directory: /user/root/flights/raw\n")
        handle.write("No files were downloaded or staged; C2-C5 were not run.\n")

print("PART_C_GATE=" + ("AVAILABLE" if available else "STOPPED"))
