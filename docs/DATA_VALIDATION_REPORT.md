# Actual-data validation report

Validated on 2026-10-30 with Spark 3.5.9 on YARN. The machine-readable evidence is generated at `results/data_validation/data_validation_report.json` by `src/00_validate_data.py`.

## Result

Status: **PASSED**. No truncation, duplicate full rows, unreadable Parquet parts, malformed holiday rows, or join multiplication was found.

| Check | Observed result |
|---|---:|
| Parquet part files | 7, all readable, Snappy-compressed |
| Rows | 1,781,482 |
| Columns | 27 |
| Date range | 2024-10-01 through 2024-12-31 |
| Distinct dates | 92 |
| Minimum / maximum daily rows | 11,212 / 21,764 |
| Distinct full-row hashes | 1,781,482 |
| Duplicate rows detected | 0 |
| Eligible modelling rows | 1,765,279 |
| Cancelled rows | 13,064 |
| Diverted rows | 3,139 |
| Holiday-calendar rows | 20 unique valid dates |
| Flights matching a holiday-calendar date | 381,195 |
| Rows after the left join | 1,781,482 |

Eligibility excludes cancelled and diverted flights and then requires `ArrDel15` to be 0 or 1. All 16,203 null target labels belong to the excluded cancellation/diversion population; there are no missing or invalid targets among eligible rows.

## Null audit

All schedule fields used by the model have zero nulls: `FlightDate`, `Reporting_Airline`, `Origin`, `Dest`, `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance`, `DayOfWeek`, and `DepTimeBlk`. Outcome fields have the expected operational null patterns:

| Column(s) | Null rows |
|---|---:|
| `CancellationCode` | 1,768,418 |
| `DepDelay`, `DepDelayMinutes` | 12,650 each |
| `ArrDelay`, `ArrDel15` | 16,203 each |
| Each delay-cause field | 1,493,518 |

These outcome and delay-cause fields are not predictors.

## Chronological modelling population

| Split | Dates | Eligible rows | Delayed 15+ min | Prevalence |
|---|---|---:|---:|---:|
| Train | 2024-10-01–2024-11-24 | 1,067,747 | 146,735 | 13.7425% |
| Validation | 2024-11-25–2024-12-03 | 173,549 | 29,526 | 17.0131% |
| Test | 2024-12-04–2024-12-31 | 523,983 | 111,703 | 21.3181% |

The rising prevalence demonstrates temporal shift and is why validation-selected thresholds, PR-AUC, and class-1 metrics are reported instead of relying on accuracy or a default 0.5 threshold.

## Holiday-distribution shift

Holiday categories do not overlap across the three windows: Halloween appears in training, Thanksgiving week appears in validation, and Christmas/New Year appears in test. Consequently, category-specific holiday coefficients cannot be interpreted as stable or causal effects. The final report treats holiday results as observational and includes separate holiday/non-holiday test slices.

## Reproduce

```powershell
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client /src/00_validate_data.py'
```

The validator exits non-zero if core schema fields are missing, input rows are duplicated/truncated by the join, date coverage is discontinuous, eligibility is inconsistent, or holiday dates are invalid or duplicated.
