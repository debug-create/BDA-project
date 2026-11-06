# Navigation, widget and copy changes

## Final routes

| Route | Sidebar | Result |
|---|---|---|
| `#/overview` | Yes | Question, selected-model headline, Season Strip, pipeline strip and result links |
| `#/operations` | Yes | Pipeline, topology, HDFS paths, Spark settings and dated cluster snapshot |
| `#/holidays` | Yes | Holiday/regular comparison, daily line, available breakdown tabs and descriptive delay-cause copy |
| `#/models` | Yes | Selected and reference models, data-backed ranking controls, feature importance and two holiday rows |
| `#/method` | Yes | Split timeline and pre-departure input list |
| `#/data` | No | Direct-link dataset summary |

The Evidence Register and About/Viva routes and navigation items were removed.

## Widgets hidden

- Other comparison-model cards
- Deep-model card
- Ablation results
- Full per-slice model table
- Evidence Register
- About and viva content
- PR and ROC curve widgets when `curves` is absent
- Confusion matrix when `confusion` is absent
- Airline breakdown tab when the airline result array is empty

The optional curve, matrix and airline widgets do not show replacement panels. They appear when their JSON fields contain data.

## Strings removed from the rendered site

- `Evidence Register`
- `About & viva`
- `Gate board`
- `Frozen`, `freeze` and related selection wording
- `Candidate`
- Design-generation labels and `G1` / `G2` / `G3`
- `Pending`, `failed`, `sample`, `killed` and `not eligible`
- `Accuracy` and `F1`
- `Validation` and standalone `Test`

The site now uses `Tuning window`, `December test window`, `Selected model`, `Reference model`, `Runs on`, `Base rate` and `× the base rate`.

## Data binding

- `model_runs.json` supplies the selected and reference run metrics.
- The primary AUC-ROC value selects the run with `role: "selected"` and `split: "test"`.
- AUC-ROC uses the JSON display precision and renders as a decimal.
- PR-AUC, base rate, top-10% precision and lift remain tied to the same run record.
- Runtime labels map from each run's `executed_on` field.
- No result value is typed directly into a React component.
