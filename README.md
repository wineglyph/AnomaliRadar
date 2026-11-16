# AnomaliRadar

**State-aware, empirically calibrated anomaly detection on the MetroPT-3 air-production compressor.**

AnomaliRadar is a reproducible research-engineering release: a deterministic data
layer, two complementary anomaly scores, empirical Mondrian calibration, a
calibration-only alarm-budget solver, a 30-minute alarm debounce, four baselines,
a pre-registered F1–F9 falsification audit, and a read-only Streamlit viewer for
the frozen results.

The scientific implementation, experiment outputs, metrics and F1–F9 verdicts
are frozen. The release adds presentation, packaging and documentation; it does
not retrain, retune or redesign the detector.

> **Status: CONDITIONAL GO.** At the primary budget of **1 alarm per observed
> day**, the recommended OR method detects **2 of 4** labelled events with **1
> false alarm** across the four event folds. It misses events 1 and 2. The
> independent unlabelled tail realizes **0.5609 alarms/day at a 1/day request
> (−43.9%)**. These are real, small-sample results, not a deployment guarantee.

---

## Problem

Industrial air-production compressors fail in ways that are not always visible
in a single sensor value. Some faults appear as an unusual value while the
machine is loaded; others appear as an abnormal persistence in a known operating
state. A detector that ignores operating context can confuse a normal state
transition with a fault. A detector that scores every minute without a calibrated
alarm layer can also produce an unusable alarm volume.

AnomaliRadar addresses four questions on one real dataset:

1. Can a **value-based** score and a **state-persistence** score be composed so
   that each contributes a detection the other misses?
2. Can the alarm threshold be selected **without looking at the test window**?
3. Can the realised alarm rate be compared against a requested budget rather
   than reported as an uncalibrated score threshold?
4. Can the result survive a pre-registered falsification audit and an honest
   limitations statement?

The contribution is an **auditable engineering composition** of established
ideas — context-aware residual scoring, state-run survival analysis, empirical
Mondrian calibration, back-off/refusal rules, and budgeted alarm selection. It
does **not** claim newly invented algorithms.

---

## Architecture

```mermaid
flowchart TD
    A[UCI MetroPT-3 CSV<br/>1,516,948 native rows] --> B[Deterministic loader<br/>schema + timestamp checks]
    B --> C[1-minute grid<br/>≤2-min forward fill]
    C --> D[Quality masks M1–M3<br/>frozen sensors, gaps, warm-up]
    C --> E[Hysteresis-confirmed state<br/>off / offloaded / loaded / other]
    E --> F[Lagged context features<br/>state, run length, transitions, day type, hour]
    D --> G[Chronological folds<br/>2/3 reference · 1/3 selection]
    F --> H[Score A: hybrid<br/>causal ridge residual nonconformity]
    E --> I[Score B: B1<br/>state-run survival nonconformity]
    G --> J[Empirical Mondrian calibration<br/>cell → state → global, n_min=200, n_refuse=50]
    H --> J
    I --> J
    J --> K[OR fusion<br/>min(pA, pB)]
    K --> L[Budget solver<br/>calibration rows only]
    L --> M[30-minute debounce<br/>accepted alarms]
    M --> N[Frozen artefacts<br/>reports/experiments]
    N --> O[Streamlit app<br/>read-only viewer]
```

**Data flow at a glance:** native ~10 s samples → 1-minute grid → masks and
confirmed state → lagged context → chronological folds → Score A and Score B →
empirical calibration → fusion → budgeted alarm selection → evaluation.

---

## Methodology

### Data layer

- **1-minute grid.** Native sampling is ~10 s with jitter. Analog channels use
  the mean of samples in the minute; digital channels use the last sample.
  Missing minutes stay missing. Forward-fill is allowed only across a gap of at
  most two minutes; larger gaps reset the segment.
- **Masks M1–M3.** M1 marks maximal runs where all seven analog channels are
  constant for at least 360 native rows (≈1 h). M2 marks native gaps longer than
  one hour. M3 marks the first 30 minutes after every M1 block or M2 gap. Masks
  are label-free retrospective quality filters, excluded from fitting,
  calibration and both numerator and denominator of alarm rates.
- **Operating state.** A 3-minute hysteresis confirms `off`, `offloaded`,
  `loaded` or `other`. `other` is a transition artefact and is excluded from
  fitting and calibration.
- **Chronological folds.** Each labelled event gets a fold whose training,
  calibration and test windows precede or contain the event. The tail fold
  reuses the frozen fold-4 model and never refits.

### Score A — context-aware value anomaly

Score A uses five value channels: `Reservoirs`, `TP3`, `H1`, `DV_pressure` and
`Oil_temperature`. Regime-defining channels and `TP2` are excluded. Per channel,
a causal Ridge model (`alpha=1.0`) is fitted on lagged values and rolling
features (lags 1/5, rolling means 15/60/360 min, rolling MAD 60 min). The
nonconformity is

```text
a_c,t = log1p( |residual_c,t| / max(causal EWMA scale_c,t, train median |residual_c|) )
a_t   = max_c a_c,t
```

Four context arms are compared: `none`, `calendar`, `state`, and `hybrid`
(lagged state + run length + transitions + day type + hour). The primary arm is
`hybrid`.

### Score B — state-persistence anomaly

Score B is a function of the confirmed state and run length only. It never reads
a value channel. Completed state runs form a survival reference; if a state has
at least 50 completed reference runs, its own reference is used, otherwise the
reference is pooled. Runs cannot bridge a data gap. The primary variant is `B1`.

### Calibration and alarm layer

- The earliest **2/3** of eligible calibration rows build the reference; the
  latest **1/3** selects `alpha`. Selection never uses test rows.
- Mondrian cells are `(lagged state, day type)`, with back-off to state and then
  global. A pool needs `n ≥ 200`; `50 ≤ n < 200` is used and flagged low-support;
  `n < 50` refuses (`p=1`, ineligible).
- The budget solver selects `alpha` on calibration data only. The 30-minute
  debounce is separate from the 3-minute state hysteresis.
- Fusion is compared under the identical alarm layer: Score A, Score B, OR
  (`min(pA,pB)`), AND (`max`), and weighted `λ=0.5`. OR is the primary.

### Evaluation

- **Recall** is the number of detected labelled events out of four.
- **Delay** is the first accepted alarm minus the labelled event start; negative
  means early.
- **Alarm rate** is alarms divided by observed days, not calendar days.
- **Budget error** is realized rate minus requested rate; the headline summary
  uses the mean absolute error over event folds.
- **F1–F9** are the pre-registered falsification criteria in
  [`docs/experimental_protocol.md`](docs/experimental_protocol.md). A `PASS` means
  the specified failure did not occur; it is not proof of correctness.

---

## Dataset and licence

| Field | Value |
| --- | --- |
| Dataset | MetroPT-3 |
| Repository | UCI Machine Learning Repository, ID 791 |
| DOI | [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R) |
| Source | <https://archive.ics.uci.edu/dataset/791/metropt+3+dataset> |
| Direct download | <https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip> |
| Licence | **CC BY 4.0** |
| Archive SHA-256 | `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a` |
| Archive size | 218,381,995 bytes (≈208 MB) |
| Native rows | 1,516,948 |
| Grid minutes | 306,960 |
| Observed minutes | 252,720 |
| Labelled events | 4 |

**Attribution.** Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021).
*MetroPT-3 Dataset* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5VW3R>.

**Redistribution.** The dataset is CC BY 4.0 but is **never committed to this
repository**. The download script fetches the official UCI archive, verifies it
against the pinned SHA-256 above, extracts the CSV and writes a small manifest.
All figures and tables derived from the data retain the attribution above.

The code in this repository is MIT-licensed; the dataset remains under CC BY 4.0.

---

## Installation

Requires **Python 3.11+**. The frozen experiment was produced with Python
3.14.10, NumPy 2.5.4, pandas 2.3.3 and scikit-learn 1.9.1. The app was smoke-tested
with Streamlit 1.67.0.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

`.[app,dev]` installs the scientific stack, Streamlit and pytest. If you only
need the experiment and tests, use `pip install -e ".[dev]"`.

### Download the dataset

```bash
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

The first command downloads the official archive, verifies the pinned SHA-256,
extracts `data/raw/metropt3/MetroPT3(AirCompressor).csv` and writes
`data/raw/manifest.json`. The second re-hashes the archive and checks both the
pinned and recorded checksums. `data/` is git-ignored and is not part of the
source archive.

---

## Reproduce the experiment

From the repository root, with the dataset downloaded:

```bash
python scripts/run_experiments.py --output reports/experiments
python scripts/make_figures.py
python -m pytest -q
```

The experiment is deterministic for a fixed configuration and dataset
fingerprint. A second full run was byte-identical for fold results, alarms,
summary, falsification and all four prediction files.

Key identity values:

- Config hash: `06954d34ced38410`
- Dataset fingerprint: `420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed`
- Falsification: **9 PASS, 0 FAIL**, `n_falsified=0`

The frozen artefacts are:

```text
reports/experiments/
  run_manifest.json
  alarms/alarms.csv
  metrics/*.csv, metrics/*.json
  predictions/fold_{1..4}.csv
  figures/*.png
```

### Run the Streamlit app

```bash
streamlit run app.py
```

The app opens six pages: **Overview**, **Data Explorer**, **Anomaly Detection**,
**Alarm Details**, **Model Comparison** and **Limitations**. Metric pages read
only `reports/experiments`. The Data Explorer additionally reads the local
MetroPT-3 CSV for sensor traces, state timelines, masks, failure windows and
gaps. If the CSV is absent, the app says so and shows the frozen artefact
evidence instead; it never substitutes synthetic data.

---

## Measured results

### Event folds at 1 alarm / observed day

The four event folds contain **7.746 observed days** in total.

| Method | Recall | Events | Alarms | False | Alarms/day | False/day | Mean abs. budget error |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Score A hybrid | 1/4 | 3 | 2 | 0 | 0.258 | 0.000 | 0.803 |
| Score B B1 | 1/4 | 4 | 9 | 1 | 1.162 | 0.129 | 1.842 |
| **OR** | **2/4** | **3, 4** | **11** | **1** | **1.420** | **0.129** | **1.644** |
| Weighted λ=0.5 | 0/4 | — | 5 | 3 | 0.646 | 0.387 | 0.704 |
| AND | 0/4 | — | 2 | 1 | 0.258 | 0.129 | 0.798 |
| Global robust z | 2/4 | 2, 3 | 9 | 7 | 1.162 | 0.904 | 1.039 |
| Rolling robust z | 0/4 | — | 7 | 7 | 0.904 | 0.904 | 1.543 |
| IForest raw | 1/4 | 2 | 12 | 6 | 1.549 | 0.775 | 1.419 |
| IForest + context | 1/4 | 4 | 22 | 10 | 2.840 | 1.291 | 3.780 |

- Event 3: Score A alarms **10 minutes before** the labelled start.
- Event 4: Score B alarms at **+32 min**; OR's first accepted alarm is **+39 min**
  because the union threshold is stricter than B's standalone threshold, not
  because of the debounce.
- Events 1 and 2: missed by A, B, OR, weighted and AND. Global robust z catches
  event 2; no primary method does.

### Independent tail at 1 alarm / observed day

The tail is 31 wall days and **24.958 observed days** with no labels. Every tail
alarm is false by construction.

| Method | Requested/day | Realized/day | Relative error | Alarms |
| --- | ---: | ---: | ---: | ---: |
| **OR** | 1.0 | **0.5609** | **−43.9%** | 14 |
| Score A hybrid | 1.0 | 1.5225 | +52.3% | 38 |
| Score B B1 | 1.0 | 1.4023 | +40.2% | 35 |
| Weighted λ=0.5 | 1.0 | 1.4825 | +48.2% | 37 |
| AND | 1.0 | 1.5225 | +52.3% | 38 |
| Global robust z | 1.0 | 0.6811 | −31.9% | 17 |

OR is inside the pre-registered 50% tail limit, with a **6.1 percentage-point
margin**. It is nowhere near a 10% guarantee.

### F1–F9

| ID | Criterion | Verdict |
| --- | --- | --- |
| F1 | Union has no unique detections | PASS |
| F2 | Union does not improve over the best single score | PASS |
| F3 | State conditioning normalizes the fault | PASS |
| F4 | Budget not controllable on tail (≤2/day, >50%) | PASS |
| F5 | >25% of alarms have calibration `n < 200` | PASS |
| F6 | Masks destroy labels (>10% of any event) | PASS |
| F7 | Scores are redundant (ρ > 0.8 and F1 holds) | PASS |
| F8 | Union loses to context-IForest at every budget with better error | PASS |
| F9 | Hygiene changes results by ≥2 events | PASS |

**9 PASS, 0 FAIL.** Several passes are narrow or low-power; see
[`docs/final_evaluation.md`](docs/final_evaluation.md) and the app's Limitations
page.

---

## Limitations

These are part of the result, not fine print.

- **Only four labelled events.** Recall is a count out of four; one event moves
  it by 25 percentage points. The event folds contain only 7.746 observed days.
- **The recommended OR method detects two events.** Events 1 and 2 are missed by
  A, B and OR. Event 3 is A-only; event 4 is B-only. The union result is real but
  fragile because it depends on both unique detections.
- **The independent tail result is 0.5609 alarms/day at a 1/day request
  (−43.9%).** This is a step function, not proportional control: 0.5/day and
  1/day select the same 14 tail alarms.
- **A strict 1/day rate cannot be guaranteed.** The pre-registered F4 limit is
  50%, not 10%; the pass is method-specific to OR. Use a hard quota or report the
  empirical range if a hard rate is required.
- **Loaded-state calibration is sparse.** Fold 4 has 41 loaded reference rows
  for Score A and 30 for Score B, both below the 50-row refusal threshold. Event
  4 is detected with global calibration (`n = 13,382`), not loaded-state
  calibration.
- **Time-series observations are not exchangeable.** Calibration is empirical,
  not distribution-free. There is no confidence interval that would survive
  multiplicity and no statistical-significance claim.
- **The conclusions are dataset-specific.** One compressor, one site, one sensor
  set, four events. The configuration was selected by the pre-registered
  fallback rule, not by population-level optimisation.
- **High budgets are not certified.** At 4/day, 36.1% of Score B alarms use
  `state_low` with `n < 200`, and OR recall (2/4) is below Score B alone (3/4).
- **Event 1's detection credit is 35.2% masked.** Only 311 of 480 credit minutes
  are usable; a miss there is not a fair test of the detector.

The decision would become **NO-GO** if the requirement were to detect events 1
and 2, guarantee ≤1 alarm/day within 10%, certify loaded-state calibration,
certify per-cell false-positive rates above 1/day, or prove statistical
significance.

---

## Repository structure

```text
.
├── app.py                      # Streamlit entrypoint and navigation
├── app_lib/                    # read-only artifact loaders and UI helpers
│   ├── artifacts.py
│   └── ui.py
├── app_pages/                  # six Streamlit pages
│   ├── overview.py
│   ├── data_explorer.py
│   ├── anomaly_detection.py
│   ├── alarm_details.py
│   ├── model_comparison.py
│   └── limitations.py
├── src/anomaliradar/
│   ├── data/                   # loader, grid, masks, state, features, folds
│   └── model/                  # scores, calibration, alarm, fusion, baselines, runner
├── scripts/
│   ├── download_data.py        # official download + pinned SHA-256 verification
│   ├── run_experiments.py      # deterministic experiment CLI
│   └── make_figures.py         # deterministic figures from frozen CSVs
├── tests/                      # frozen scientific suite + app smoke tests
├── reports/experiments/        # frozen artefacts: metrics, alarms, predictions, figures
├── docs/                       # protocol, methodology, reports, release report
├── pyproject.toml              # dependencies and pytest configuration
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

The source archive excludes `data/`, `.venv/`, caches, temporary files and the
dataset archive. The dataset download and SHA-256 verification mechanism is
preserved in `scripts/download_data.py`.

---

## Testing

The original frozen suite is **141 tests**. The release adds application smoke
tests for the artifact loaders and page renderers. Run everything with:

```bash
python -m pytest -q
```

The release verification also checks:

- no exception on any of the six Streamlit pages;
- Streamlit health endpoint and startup;
- import checks for the package and app modules;
- README command check;
- dependency installation with `pip install -e ".[app,dev]"`;
- archive contents for dataset, virtual environment, caches, credentials and
  machine-specific absolute paths;
- reported metrics against the actual experiment CSVs.

See [`docs/release_report.md`](docs/release_report.md) for the exact results.

---

## Citation

If you use this software, cite the repository via `CITATION.cff`. If you use the
dataset, cite:

> Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset*
> [Dataset]. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C5VW3R>

Associated papers:

- Davari et al. (2021), *Predictive maintenance based on anomaly detection using
  deep learning for air production unit in the railway industry*, IEEE DSAA 2021.
  <https://doi.org/10.1109/DSAA53316.2021.9564181>
- Veloso et al. (2022), *The MetroPT dataset for predictive maintenance*,
  Scientific Data 9, 764. <https://doi.org/10.1038/s41597-022-01877-3>

---

## Licence

Code: **MIT** — see [`LICENSE`](LICENSE).

Dataset: **CC BY 4.0** — see the attribution above. The dataset is not
redistributed in this repository or its source archive.

---

## Documentation

- [`docs/experimental_protocol.md`](docs/experimental_protocol.md) — binding
  protocol, C1–C10 deviations, F1–F9 criteria.
- [`docs/methodology.md`](docs/methodology.md) — score, calibration and alarm
  definitions.
- [`docs/day3_report.md`](docs/day3_report.md) — implementation and real-data
  results.
- [`docs/final_evaluation.md`](docs/final_evaluation.md) — F1–F9 verdicts,
  conditions and NO-GO triggers.
- [`docs/datasets.md`](docs/datasets.md) — dataset provenance and schema.
- [`docs/release_report.md`](docs/release_report.md) — release checks, archive
  identity and publication status.

---

## Publication status

Published as a public repository at
**<https://github.com/wineglyph/AnomaliRadar>** on branch `main`, release
`v0.2.0`. The clean source archive is attached to that GitHub release. The
repository is not a mirror and no force-push was used. See
[`docs/release_report.md`](docs/release_report.md#6-publication-status) for the
verification record.
