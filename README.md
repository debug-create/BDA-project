# Big Data Analytics: Holiday Flight Delay Pipeline

This repository supports a B.Tech Big Data Analytics mini-project using BTS domestic flight data from October–December 2024. It uses Hadoop HDFS/YARN and Spark 3.5.9 on Hadoop 3.4.3.

## Current status

- Three-node-container topology: `master`, `worker1`, `worker2`.
- HDFS raw-data upload and replication checks passed on the project owner's PC.
- `src/01_clean.py` and `src/02_holiday_analysis.py` completed successfully on YARN.
- Current candidate `src/03_ml.py` completed a real-data run. See `docs/HANDOVER_TEMPORAL_HISTORY_ISSUE.md` for latest counts, metrics and the one active code-quality task.

## Data

Obtain the official monthly BTS files separately and place them locally in `data/` using these names:

- `flights_2024_10.csv`
- `flights_2024_11.csv`
- `flights_2024_12.csv`

These large raw files are intentionally excluded from Git. `data/holidays.csv` is small and tracked.

## Docker setup

Start Docker Desktop with the WSL2/Linux backend. From the `docker` directory:

```powershell
docker compose config
docker compose --progress plain build master
docker compose up -d
docker compose ps
```

For the standard production pipeline, first upload and clean the BTS data according to the project README/handover, then submit the scripts with `spark-submit --master yarn --deploy-mode client`. Do not use local mode for the assessed production jobs.

## Current issue-only test

The current contributor task is to regression-test the past-only historical delay-rate features in `src/03_ml.py`. Read `docs/HANDOVER_TEMPORAL_HISTORY_ISSUE.md` first. The synthetic test does not need the raw BTS CSVs:

```powershell
docker exec -it master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'
```

Expected final line: `TEMPORAL_HISTORY_TEST_STATUS=PASSED`.

## Important exclusions from Git

Do not commit raw BTS CSVs, ZIP archives, cleaned Parquet, HDFS/Docker volumes, generated model artifacts, Spark/Hadoop logs, virtual environments, secrets, or the full generated `results/` folder. See `.gitignore`.
