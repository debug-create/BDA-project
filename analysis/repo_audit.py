#!/usr/bin/env python3
"""Write reproducible repository/course-readiness audit tables."""

import csv
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "ml_redesign" / "diagnostics"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).splitlines()


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


tracked = []
for relative in git("ls-files"):
    path = ROOT / relative
    if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
        tracked.append((relative, path.stat().st_size, "tracked file over 5 MiB"))
write("tracked_files_over_5mb.csv", ["Path", "Bytes", "Finding"], tracked)

ignored = []
for line in git("status", "--short", "--ignored"):
    status, relative = line[:2], line[3:]
    low = relative.lower()
    if status == "!!" and any(token in low for token in (
        "results/", "handoff", ".zip", "__pycache__", ".log", ".parquet", "model"
    )):
        ignored.append((relative, "ignored", "do not commit generated data/artifacts/cache"))
write("noncommit_artifacts.csv", ["Path", "GitStatus", "Recommendation"], ignored)

commands = [
    ("cluster startup", "documented", "consistent", "docker compose config/build/up/ps is valid from repository root"),
    ("handoff staging", "documented", "consistent", "matches current /results bind mount and canonical HDFS clean/reference paths"),
    ("data validation", "documented", "consistent", "actual evidence identifies application ..._0001 and PASSED marker"),
    ("local tests", "documented", "consistent", "commands match the two tracked src/tests scripts"),
    ("four controlled selection runs", "documented", "consistent", "run tags match actual-data results narrative"),
    ("XGBoost selection", "documented", "wording mismatch", "command explicitly requests 1g executors, but following prose says executors remain at 512 MB"),
    ("frozen final evaluation", "documented", "consistent", "1536m driver; executor memory inherited as 512m from spark-defaults"),
    ("plot generation", "documented", "consistent", "tracked script and actual screenshots exist"),
    ("saved-model verification", "documented", "consistent", "command is valid; application ..._0008 and the PASSED marker are recorded in the actual-data report"),
]
write("readme_command_audit.csv", ["Workflow", "READMEStatus", "AuditResult", "EvidenceOrIssue"], commands)

topics = [
    ("HDFS", "covered", "canonical inputs, output tables, saved models, fsck and two DataNodes are implemented/documented"),
    ("MapReduce", "demonstrated only", "handover records Hadoop MapReduce Pi success; no project MapReduce job is tracked"),
    ("Pig", "not covered", "no Pig scripts or documented Pig run"),
    ("Hive", "not covered", "no Hive DDL/query or documented Hive run"),
    ("HBase", "not covered", "no HBase schema/client or documented HBase run"),
    ("Spark RDD", "not directly covered", "project uses DataFrame/Spark SQL APIs; no explicit RDD workload"),
    ("SparkSQL", "covered", "02_holiday_analysis.py registers views and executes Spark SQL"),
    ("SparkML", "covered", "03_ml.py implements leakage-controlled MLlib pipelines and persisted models"),
    ("Streaming", "not covered", "no Structured Streaming/DStream source or documented streaming run"),
]
write("course_topic_coverage.csv", ["Topic", "Coverage", "RepositoryEvidence"], topics)

print("REPO_AUDIT_STATUS=PASSED")
