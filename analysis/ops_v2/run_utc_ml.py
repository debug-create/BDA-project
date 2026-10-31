#!/usr/bin/env python3
"""Run the existing ML stage with the audited UTC as-of implementation.

This adapter leaves ``src/03_ml.py`` unchanged.  It replaces only its as-of
feature attachment hook, preserving the existing split, tuning, freezing,
threshold, reporting, and held-out-test controls.
"""

import importlib.util
import json
import sys

if "/src" not in sys.path:
    sys.path.insert(0, "/src")

import asof_ops_utc


spec = importlib.util.spec_from_file_location("ml_stage", "/src/03_ml.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


def attach_utc(event_source, eligible, lead_minutes, prior_cutoff, keep, release, priors=None):
    timezone_table = keep(asof_ops_utc.load_timezone_offsets(event_source.sparkSession))
    timezone_table.count()
    result, fitted_priors, coverage = asof_ops_utc.attach_asof_features_utc(
        event_source, eligible, lead_minutes, prior_cutoff, timezone_table,
        keep, release, priors=priors)
    release(timezone_table)
    print("ASOF_UTC_COVERAGE=" + json.dumps(coverage, sort_keys=True), flush=True)
    return result, fitted_priors


ml.attach_asof_features = attach_utc
ml.REQUIRED_EVENT_COLUMNS = asof_ops_utc.REQUIRED_UTC_EVENT_COLUMNS

print("DESIGN_EVALUATION_LABEL=THIRD DESIGN EVALUATION, UTC-corrected", flush=True)
raise SystemExit(ml.main())
