# Handover: Verify and protect the Spark MLlib temporal-history feature fix

**Project:** Big Data Analytics Mini Project — Quantifying Holiday Effects on U.S. Flight Delays and Cancellations (Oct–Dec 2024)  
**Course context:** BDA, Semester V  
**Owner of main project:** Pragya  
**Contributor scope:** one narrow issue only — audit/test the temporal historical-rate features in `src/03_ml.py`.

## 1. What the contributor should do

Do **not** rebuild the ML pipeline from scratch or change the project topic. Start from the current `src/03_ml.py` in the shared repository. Review and regression-test the historical delay-rate feature logic. Fix `src/03_ml.py` only if a test exposes a defect; add/maintain the focused regression test in `src/tests/test_temporal_history_features.py`.

The goal is to prove that a training row dated **D** cannot use an outcome label from the same date or from any date after D when constructing historical-rate features.

### Acceptance criteria

1. For training rows on date D, group rates use labels from dates strictly earlier than D.
2. The smoothed global prior used for a training row on D also uses only data dated before D. The first date has no history and uses the script's documented neutral fallback of 0.5.
3. Changing labels on date D or a future date must not change historical features for a row dated D.
4. Changing labels on a previous date may change historical features for a later date.
5. The history-feature joins do not change the number of flight rows or duplicate a flight ID in the regression fixture.
6. Validation and test features are derived only from the completed earlier fitting window: Stage A rates from the training window; Stage B/test rates from train + validation. Do not use validation/test outcome labels to construct test features.
7. Keep the existing file name `03_ml.py`, HDFS paths, output filenames, target, eligibility rules, chronological split and YARN run command unless a specific defect proves a change is necessary.
8. Add no model families, feature expansions, hyperparameter tuning or threshold changes as part of this issue. Those are separate experiments.
9. Provide test output and a pull request that explains what was checked. Do not claim the full data run passed unless it was actually run.

## 2. The issue and its history

### Earlier defect

An earlier implementation aggregated historical rates across the entire fitting window and then subtracted only the current flight date (`leave-one-day-out`). That removed same-day labels, but still let an early training date use labels from later dates inside the fitting window. Example: an October row could receive a rate partly calculated from November flight outcomes. This is temporal leakage even if each row's own label is removed.

### Current candidate fix

The current script in this packet includes `apply_history_training_past_only()` and calls it for fitting rows from `prepare_stage()`. It calculates per-key daily aggregates, then uses a window ordered by `FlightDate` with rows strictly preceding the current date. It applies the same past-only rule to the global prior; for the first date it uses 0.5 because there is no earlier history.

For validation/test rows, `apply_history()` joins rates from the completed earlier fitting window. Stage A uses train to create validation features; Stage B fits on train + validation and applies those final fitting-window rates to the held-out test.

**Status:** the current candidate script has run successfully on the real dataset. That run confirms execution, not by itself the absence of leakage. The regression test is the outstanding safeguard. The contributor should inspect the actual branch version instead of assuming this handover's description is sufficient.

## 3. Current project status — do not restart from zero

The main project's infrastructure and upstream jobs are already complete:

- Docker Desktop with WSL2 backend on Windows.
- Hadoop 3.4.3, Spark 3.5.9, Java 11 in the Docker image.
- Containers: `master`, `worker1`, `worker2`.
- HDFS reported 2 live DataNodes, replication factor 2, and healthy blocks.
- YARN reported 2 RUNNING NodeManagers.
- Hadoop MapReduce Pi succeeded.
- Spark Pi succeeded on YARN and logged tasks on both worker hosts.
- Raw files uploaded to `/user/root/flights/raw/`: `flights_2024_10.csv`, `flights_2024_11.csv`, `flights_2024_12.csv`. `hdfs fsck` reported HEALTHY, 7 blocks, replication 2, no missing/corrupt/under-replicated blocks.
- `01_clean.py` succeeded: 1,781,482 rows retained; zero rows dropped for invalid/missing date/core route fields. Cleaned Parquet is at `hdfs://master:9000/user/root/flights/clean`.
- `02_holiday_analysis.py` succeeded and wrote its aggregate outputs under `/user/root/flights/results/holiday_analysis/` and `C:\bda\results\holiday_analysis\`.
- The current `03_ml.py` completed a real-data run successfully and saved metrics/models. The current candidate's observed metrics are listed below.

### Data and split counts from the latest candidate run

| Item | Count / value |
|---|---:|
| Cleaned records | 1,781,482 |
| Cancelled flights excluded from ML | 13,064 |
| Diverted flights excluded from ML | 3,139 |
| Eligible ML records | 1,765,279 |
| Training rows | 1,067,747 (2024-10-01 through 2024-11-24) |
| Validation rows | 173,549 (2024-11-25 through 2024-12-03) |
| Test rows | 523,983 (2024-12-04 through 2024-12-31) |

### Latest candidate run metrics (diagnostic baseline; not a guarantee of the final numbers)

| Test metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Threshold selected on validation F1 | 0.10 | 0.13 |
| AUC-ROC | 0.62232 | 0.60256 |
| AUC-PR | 0.29950 | 0.28248 |
| F1 for delayed class | 0.37788 | 0.36947 |
| Precision for delayed class | 0.24473 | 0.25297 |
| Recall for delayed class | 0.82880 | 0.68489 |
| Accuracy | 0.41825 | 0.50167 |

These results indicate modest predictive signal. The delayed-flight rate increases over time: 13.74% training, 17.01% validation, 21.32% test. Do not tune using the held-out test period; do not promise an accuracy improvement as part of this issue.

## 4. Relevant project paths and environment

Assumed repository layout when cloned to `C:\bda`:

```text
C:\bda\
  docker\
    Dockerfile
    docker-compose.yml
    entrypoint.sh
    conf\
  data\
    holidays.csv
    flights_2024_10.csv       # local only; do NOT commit
    flights_2024_11.csv       # local only; do NOT commit
    flights_2024_12.csv       # local only; do NOT commit
  src\
    01_clean.py
    02_holiday_analysis.py
    03_ml.py
    tests\
      test_temporal_history_features.py
  results\                  # generated/local; generally do NOT commit
  screenshots\               # selected report evidence may be committed
  docs\
```

Versions in the supplied Docker image: Ubuntu 22.04, Hadoop 3.4.3, Spark 3.5.9, OpenJDK 11, Python 3.10 from Ubuntu 22.04, with `python3-numpy` and `python3-pandas` installed. NumPy and pandas must exist on the driver and worker containers because MLlib/PySpark use them. Spark configuration requests two executors at 512 MB each and `spark.sql.shuffle.partitions=8`.

Paths used by the ML job:

- Clean Parquet: `hdfs://master:9000/user/root/flights/clean`
- Holiday calendar: `hdfs://master:9000/user/root/flights/reference/holidays.csv`
- ML tables: `hdfs://master:9000/user/root/flights/results/ml`
- Models: `hdfs://master:9000/user/root/flights/models`
- Windows-mounted results: `C:\bda\results\ml` (container path `/results/ml`)
- Script inside container: `/src/03_ml.py`

## 5. How the contributor should test on their own device

### Recommended: run the small regression test only

This task does **not** require the 769 MB raw dataset or a complete run of all three jobs. The test uses a six-row synthetic fixture and checks the temporal invariant. This is enough for local code review, but it does not replace the final integration run on the project owner's existing dataset/cluster.

Prerequisites: Windows 10/11, Docker Desktop with the WSL2/Linux backend, internet for the first image build, and a few GB of free disk space. No Windows installation of Hadoop, Spark or Java is required; the container image supplies them.

1. Clone the shared repository into `C:\bda` (or ensure the same relative folder layout):

   ```powershell
   git clone <REPOSITORY-URL> C:\bda
   Set-Location C:\bda\docker
   ```

2. Check Docker Desktop is running, then validate/build/start:

   ```powershell
   docker compose config
   docker compose --progress plain build master
   docker compose up -d
   docker compose ps
   ```

   The first image build downloads the Hadoop and Spark archives and can take several minutes. The Compose file builds the shared image once through `master`; the workers reuse it. No raw flight data is required for the regression test.

3. Wait for the three containers to start, then run the test **in local Spark mode** (local mode is only for this small unit test, not for the assessed production pipeline):

   ```powershell
   docker exec -it master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'
   ```

4. Expected final line:

   ```text
   TEMPORAL_HISTORY_TEST_STATUS=PASSED
   ```

   This test checks the first-day fallback, same-day exclusion, future-day exclusion, that prior-day labels can affect later rates, and row-count preservation.

5. If it fails, save the full traceback and the command used. Do not change the full pipeline or invent a passing result. Make the smallest fix in `src/03_ml.py`, rerun the test, and explain the change in the PR.

### Full integration run (only if specifically requested)

A fresh device will not have the named HDFS volume/data. A full run requires the three original CSVs in `C:\bda\data`, the cluster started, data ingested/cleaned, the reference calendar in HDFS and then the production job. This is intentionally out of scope for the issue-only contributor unless the project owner explicitly requests a full independent reproduction. The final end-to-end run should be performed by the owner on the established project cluster after the merge.

Production command on the project's main cluster:

```powershell
docker exec -it master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 512m --executor-memory 512m --num-executors 2 --executor-cores 1 /src/03_ml.py'
```

Do not substitute `--master local[*]` for the assessed run. The unit test uses local mode only to test the feature helper cheaply.

## 6. GitHub files and branch workflow

### Track in GitHub

Commit the project source and reproducibility files:

- `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/entrypoint.sh`, `docker/conf/*`
- `src/01_clean.py`, `src/02_holiday_analysis.py`, `src/03_ml.py`
- `src/tests/test_temporal_history_features.py`
- `data/holidays.csv`
- `README.md`, `.gitignore`, `docs/HANDOVER_TEMPORAL_HISTORY_ISSUE.md`
- a **curated** subset of small screenshots/results needed for the report, if desired

Do **not** commit raw flight CSVs (each is hundreds of MB), ZIP archives, cleaned Parquet, HDFS/Docker named volumes, Spark/Hadoop logs, model binaries, virtual environments, credentials, or the full generated `results/` tree. GitHub has a 100 MB hard limit for regular files; the raw CSVs exceed it. Keep raw data on each machine under `C:\bda\data` and describe where it comes from in the README. Do not upload any personal tokens, secrets or user-specific `.wslconfig` files.

Use `.gitignore` from this packet as a starting point. Review `git status` before pushing; a `.gitignore` does not remove a file already tracked.

### First GitHub push (project owner; only if this folder is not already a Git repository)

Open PowerShell at `C:\bda`. If the repository has already been initialized, skip `git init` and do not replace the existing remote.

```powershell
git status
# Only if this folder is NOT already a Git repository:
git init -b main

git add .gitignore README.md docker src docs data/holidays.csv screenshots/.gitkeep results/.gitkeep
git status
git diff --cached --stat
# Check that no flights_2024_*.csv, results tables, or generated artifacts are staged.
git check-ignore -v data/flights_2024_10.csv

git commit -m "chore: add BDA Hadoop Spark pipeline and ML baseline"
# Only if origin has not already been configured:
git remote add origin <REPOSITORY-URL>
git push -u origin main
```

Replace `<REPOSITORY-URL>` with the URL of the dedicated BDA repository. If `git remote -v` already shows the correct `origin`, do not run `git remote add origin` again. Do not push until `git status` confirms that the large raw CSVs and generated results are not staged.

### Suggested collaboration flow

1. Owner commits the known-good project state and the current candidate `03_ml.py` to `main`.
2. Contributor branches from that exact commit:

   ```powershell
   git switch -c fix/temporal-history-leakage
   ```

3. Contributor works only in `src/03_ml.py` and `src/tests/test_temporal_history_features.py`, unless a test-only documentation adjustment is needed.
4. Contributor runs the synthetic regression test, commits and pushes:

   ```powershell
   git add src/03_ml.py src/tests/test_temporal_history_features.py
   git commit -m "test(ml): guard temporal history features against leakage"
   git push -u origin fix/temporal-history-leakage
   ```

5. Open a pull request. Include: the bug being tested, the exact test command, its real output, files changed, and any remaining limitations. Do not merge without a review.
6. Owner merges only after review, then runs the production `spark-submit --master yarn` job on the existing cluster and preserves the resulting metrics separately from the previous run.

If the repository is not created yet, create a dedicated repository for this BDA mini-project; do not mix these files into an unrelated project repository.

## 7. Suggested `.gitignore`

The packet's `.gitignore` ignores raw BTS files, generated results, model artifacts, logs and Python cache files while allowing the small holiday calendar to be tracked. Review it before the first commit.

## 8. What to return to the owner

- Pull request link or patch/diff.
- Test command and unedited test output.
- Short explanation of how the test proves same-day/future labels cannot change a training row's history features.
- Confirmation of whether the full job was run (usually it will not be run on the contributor device).
- No new model-performance claims unless the complete real-data pipeline was actually rerun and the held-out metrics were saved.
