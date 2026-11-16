# Day 0 Report — MetroPT-3 Validation and EDA

**Date:** 2026-11-16
**Scope:** data validation, licensing, EDA. No models, no detectors, no CCCS, no app.
**Reproduce with:** `python scripts/download_data.py && python scripts/eda_day0.py`
**Figures:** `reports/day0/figures/` · **Machine-readable:** `reports/day0/summary.json`,
`failure_windows.csv`, `context_cells.csv`

---

## 1. Dataset inspection

### 1.1 Download and integrity

| Item | Value |
| --- | --- |
| URL | `https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip` |
| Size | 218,381,995 bytes |
| SHA-256 | `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a` |
| Archive members | `MetroPT3(AirCompressor).csv`, `Data Description_Metro.pdf` |
| License | CC BY 4.0 (verified on the UCI page) |

Downloaded, extracted and checksummed on 2026-11-16. Manifest in `data/raw/manifest.json`.

### 1.2 Actual dimensions and schema

| Metric | Value |
| --- | --- |
| Rows | **1,516,948** |
| Columns in file | 17 (1 row counter + `timestamp` + 7 analog + 8 digital) |
| Data columns | 15 sensors |
| Time range | **2020-02-01 00:00:00 → 2020-09-01 03:59:50** |
| Span | 213.17 days |
| Missing cells | **0** in the entire file |
| Duplicate timestamps | **0** |
| Duplicate full rows | **0** |
| Timestamp monotonic | yes |
| Timestamp parse failures | 0 |

Two documentation errors were found and are recorded in `docs/datasets.md`:

1. The PDF claims **15,169,480 points at 1 Hz**. The file holds **1,516,948** rows —
   exactly a factor of 10 fewer — at a **10 s** interval. The "row counter" column
   is the decisecond timestamp, which explains the 10× overstatement. True logging
   rate is 0.1 Hz, not 1 Hz.
2. Three columns are spelled differently in the CSV than in the UCI description:
   `DV_eletric`, `Towers`, `Caudal_impulses` (vs `DV_electric`, `TOWERS`,
   `Caudal_impulse`). The CSV is authoritative.

### 1.3 Sampling regularity — a real complication

| Statistic | Value |
| --- | --- |
| Median interval | 10.0 s |
| Mean interval | 12.14 s |
| Exactly 10 s | 88.2% of gaps |
| 9 s / 12 s / 13 s | 128,277 / 38,321 / 7,988 gaps |
| Gaps > 60 s | **331** |
| Gaps > 1 h | **160** |
| Largest gap | **48.03 h** |

The series is **not** a clean uniform grid. Irregularities are of two kinds, and
they have different causes:

- **Small jitter (8–13 s)** — logger scheduling. Harmless but means "row index" is
  not a time index; always resample or use elapsed time.
- **Large gaps (1–48 h)** — these cluster in the **night and early morning**
  (20:00–05:00). Row density by hour of day drops to ~70–80% of the 06:00 peak.
  This is consistent with the train being out of service, not with sensor failure.
  **Implication:** the compressor's operating envelope is systematically different
  at night, so "hour of day" is partly a *data-availability* variable as well as a
  process variable. A context model must not learn "no data at 03:00" as normal.

### 1.4 Frozen sensor block (33.2% of all rows are inside \(\ge\)1 h constant runs)

The longest identical-value run is **51.43 h**:

```
2020-06-22 15:06:11 → 2020-06-25 05:13:16
```

During that block **all seven analog channels are bit-for-bit constant** while four
digital channels keep toggling. That is a stuck/held sensor or a logging replay, not
physics. There are **7 runs longer than 1 h**, together covering **3.32%** of rows.

This block does **not** overlap any labelled failure window, so it cannot be excused
as "the fault". It must be excluded or explicitly masked before calibration,
otherwise any density- or variance-based score will treat it as maximally normal
(and inflate the normal class with 18,515 fake points).

### 1.5 Sensor-level quirks

- `TP2`, `H1`, `DV_pressure` carry a **negative zero-offset** (TP2 min −0.032,
  H1 min −0.036, DV_pressure min −0.032). Values are not strictly physical
  pressures; do not clip at 0 without recording the choice.
- `TP2` is \(\approx 0\) in **99.99% of "off" rows** and **99.47% of "offloaded"
  rows**, and non-zero only when loaded. It is effectively a regime indicator, not
  an independent pressure measurement.
- `LPS` is active in only 0.34% of rows; `start` regime is 44 rows total. Both are
  too rare to be used as context cells.
- `MPG` and `COMP` agree on 99.57% of rows, as the documentation states.

---

## 2. Context suitability

### 2.1 Operating-state variables (verified encoding)

The dataset PDF names the encoding of `Motor_current`: ~0 A off, ~4 A offloaded,
~7 A loaded, ~9 A start. Cross-tabulating current bands against `DV_eletric`
confirms it empirically:

| Band (A) | `DV_eletric`=0 | `DV_eletric`=1 |
| --- | --- | --- |
| < 0.5 ("off") | 826,795 | 2,067 |
| 2–5 ("offloaded") | 441,859 | 15,425 |
| 5–8 ("loaded") | 4,408 | 226,087 |
| ≥ 8 ("start") | 1 | 43 |

So a joint rule on current band + `DV_eletric` recovers the documented states with
only **1.32% "other"** (transition) rows. Verified regime shares:

| Regime | Share | Median episode length |
| --- | --- | --- |
| off | 54.6% | 71 rows (≈11.8 min) |
| offloaded | 29.1% | 41 rows (≈6.8 min) |
| loaded | 14.9% | 12 rows (≈2.0 min) |
| other (transition) | 1.3% | 1 row |
| start | 0.003% | 1 row |

The compressor exhibits a clean **duty cycle**: ~2 min loaded, ~7 min offloaded,
~12 min off. This is a genuine, exploitable context structure. Sensor medians
separate clearly across regimes (e.g. `Reservoirs` median 8.6 / 9.6 / 8.9 bar;
`Oil_temperature` 61 / 69 / 69 °C).

### 2.2 Calendar context

Hour-of-day, weekday/weekend, and month are all available and well populated
(212 distinct dates: 152 weekdays, 60 weekend days). Night hours overlap the
outages, as noted in §1.3.

### 2.3 Context sparsity

At the finest proposed granularity (regime × day-type × hour):

| Metric | Value |
| --- | --- |
| Cells | 222 |
| Median rows / cell | 4,793.5 |
| Smallest cell | 1 row |
| Cells with < 200 rows | **54** (24%) |
| Cells < 200 after 1-min resampling | 78 |

So three quarters of cells are comfortably dense, but a quarter are not — and they
include exactly the rare states (`start`, transitions) one would most want to
condition on. **A hierarchical back-off is not optional; it is required.** This
validates the back-off design in the proposal at the data level.

---

## 3. Label suitability — the most important finding

### 3.1 What the labels actually are

Four "air leak / high stress" windows come from the operator's failure reports.
They are **maintenance windows, not point labels and not fault-onset times**.
Two documentation problems:

- The PDF numbers the rows `#1, #1, #3, #4` (duplicate `#1`). Sequential ids are
  used here with the original label preserved as `pdf_label`.
- The PDF also lists maintenance dates (30 Apr 12:00, 8 Jun 16:00) that fall
  *between* windows, which is consistent with repair-after-failure but is not a
  label.

| # | Window (2020) | Duration | Rows | Coverage vs 10 s grid | `Reservoirs` Δ vs prior 7 d |
| --- | --- | --- | --- | --- | --- |
| 1 | 04-18 00:00 → 04-18 23:59 | 23.98 h | 8,657 | 100.3% | −0.168 bar |
| 2 | 05-29 23:30 → 05-30 06:00 | 6.50 h | 2,360 | 100.9% | −0.381 bar |
| 3 | 06-05 10:00 → 06-07 14:30 | 52.50 h | 17,315 | 91.6% | −0.963 bar |
| 4 | 07-15 14:30 → 07-15 19:00 | 4.50 h | 1,622 | 100.1% | −0.805 bar |

Total labelled time is **87.5 h = 1.71%** of the 213-day span. Class imbalance is
therefore severe at the point level, and the label is *coarser* than the phenomenon.

### 3.2 The actual failure signature is a sustained state, not a spike

Every window shows the same behaviour, and it is unambiguous:

| # | Loaded share, prior 7 d | Loaded share, window | Offloaded share, prior 7 d | Offloaded share, window |
| --- | --- | --- | --- | --- |
| 1 | 16.0% | **98.9%** | 19.7% | 0.0% |
| 2 | 11.3% | **99.1%** | 29.2% | 0.8% |
| 3 | 15.7% | **99.0%** | 29.2% | 0.0% |
| 4 | 18.8% | **97.0%** | 43.5% | 2.6% |

The compressor stops cycling and runs loaded almost continuously; reservoir pressure
sags instead of oscillating. See `figures/07_failure_*_context.png`. This is
physically coherent: a leak means demand cannot be met, so the unit never reaches
its unload setpoint.

**Consequences for the project, stated plainly:**

1. This is a **context/state anomaly**, which is exactly the project's thesis — but
   the "context" that changes is the *endogenous operating state*, not the calendar.
   A value-only detector will see near-normal pressures and miss it; a state-aware
   detector can see it immediately.
2. **Circularity risk.** Regime is derived from `Motor_current`/`DV_eletric`, which
   are the very channels that reveal the fault. Conditioning a `Motor_current`
   anomaly score on regime computed from `Motor_current` would be partly
   self-fulfilling. `TP2` is likewise ~deterministic on regime (§1.5). Any
   context-aware design must either (a) use only *exogenous* context (calendar,
   and lagged/prior-day state), or (b) treat state persistence as a **feature under
   test**, not as a conditioning variable. This must be resolved before Day 1
   modelling, and it is the main reason the verdict below is conditional.
3. **Evaluation must be event-level**, not point-level. A "detection" is an alarm
   raised during the window (with a tolerance), not a per-row classification. Point
   metrics over 1.5 M rows with 1.7% positive would be dominated by alarm volume.

### 3.3 Data available for chronological splits

| Segment | Days | Rows |
| --- | --- | --- |
| Before failure #1 (clean pre-period) | 77.0 | 562,564 |
| After failure #4 (clean tail) | 47.4 | 344,233 |
| Within/around labelled windows | 87.5 h | ~30k |

There is ample clean data for train/calibration and a genuinely held-out tail.
Rolling-origin splits are feasible: 77 days of pre-failure normal data is more
than enough to calibrate a quantile model.

---

## 4. Figures produced

| File | Content |
| --- | --- |
| `01_sensor_overview.png` | All 7 analog channels over 7 months; red = labels, grey = frozen block |
| `02_sampling_intervals.png` | Interval histogram (log y), annotated with gap counts |
| `03_missingness.png` | Missing-value bar chart — explicitly shows zero missing cells |
| `04_regime_distribution.png` | Verified regime shares with percentages |
| `05_sensors_by_regime.png` | `Motor_current`, `Reservoirs`, `Oil_temperature` by regime |
| `06_label_timeline.png` | Label timeline with the frozen block marked |
| `07_failure_{1..4}_context.png` | Each failure ±12 h: reservoir pressure, motor current, TP3 |

No figure is committed with fabricated data; every one is regenerated by
`scripts/eda_day0.py`.

---

## 5. Tests executed

Command: `python -m pytest tests/ -q` → **10 passed in 8.08 s**.

| Test | Asserts |
| --- | --- |
| `test_synthetic_fixture_passes_contract` | The 600-row synthetic fixture satisfies the full schema contract |
| `test_missing_column_is_rejected` | Dropping `Motor_current` fails with a clear message |
| `test_string_timestamp_is_rejected` | Unparsed timestamps are rejected |
| `test_unsorted_timestamps_are_rejected` | Non-monotonic time is rejected |
| `test_non_binary_digital_signal_is_rejected` | A digital channel with value 2 is rejected |
| `test_impossible_motor_current_is_rejected` | 9999 A is rejected by the plausibility range |
| `test_regime_definition_is_exhaustive` | COMP/DV combinations stay within the 4 valid pairs |
| `test_uci_page_spellings_are_normalised_to_csv_names` | UCI spellings fail the contract, alias mapping fixes them |
| `test_calibration_split_ordering_is_enforceable` | A 60/20/20 chronological split has strictly ordered, non-overlapping ranges |
| `test_real_dataset_satisfies_contract` | Runs on the real 1.5 M-row file when present; checks schema, 0 duplicate timestamps, 10 s median, binary `Towers` |

Nine tests run without any download (synthetic fixture only). The tenth is
auto-skipped in CI when the dataset is absent, so the suite stays fast and offline.

**Honest note:** the plausibility-range test initially failed on the real data
because I had assumed non-negative pressures. The negative offsets in §1.5 are a
genuine dataset property, discovered by the test, and the range was widened to
match reality rather than the data being clipped.

---

## 6. Risks and limitations

| # | Risk | Severity | Status / mitigation |
| --- | --- | --- | --- |
| 1 | **Regime is both context and symptom** (loaded share 15%→99% during every fault) | **High** | Must choose exogenous vs lagged context; do not condition on same-row regime. Blocks GO. |
| 2 | Labels are coarse maintenance windows, not onsets | High | Event-level evaluation with tolerance; report detection delay, never point-F1. |
| 3 | 51 h frozen block (3.3% of rows) | High | Mask runs ≥ 1 h before calibration; record the mask. |
| 4 | Nightly outages (160 gaps > 1 h) confound hour-of-day | Medium | Treat as missing-at-random-by-operation; never impute across night gaps. |
| 5 | Irregular sampling (88% exactly 10 s) | Medium | Resample to a fixed 1-min grid, or use elapsed-time features. |
| 6 | 24% of context cells have < 200 rows | Medium | Hierarchical back-off required (already designed). |
| 7 | Severe imbalance (1.71% labelled time) | Medium | Precision/recall at alarm budget, not accuracy. |
| 8 | Only 4 events | **High** | Statistical power is tiny; CIs will be wide. Must be stated, not hidden. |
| 9 | `TP2`/regime near-determinism | Medium | Treat as leakage; exclude TP2 from "independent" evidence. |
| 10 | Documentation errors in the official PDF | Low | Documented in `docs/datasets.md`; CSV treated as authoritative. |

Risk 1 and risk 8 are the two that actually constrain the project. Everything else
is engineering.

---

## 7. Verdict

### **CONDITIONAL GO**

MetroPT-3 is a genuine, well-licensed, richly contextual dataset with real labelled
faults, a clean duty cycle, 77 days of pre-failure normal data, and a 47-day clean
tail. It can support the project. But it cannot support the project *as originally
framed*, for three concrete reasons, each of which needs a documented protocol
change:

**Condition 1 — Resolve the context circularity before any modelling.**
The fault signature is a change in operating state, and the state variables are
also the fault indicators. Day 1 must adopt one of:
- **(a) Exogenous-only context**: calendar (hour, weekday/weekend, month) plus
  slowly-varying non-fault channels, with regime excluded from conditioning.
- **(b) Lagged state context**: state measured at \(t-\Delta\) (e.g. 1 h or
  previous day), never same-row.
- **(c) Two-score design**: a calibrated value score plus a separate
  state-persistence score, combined at the alarm layer and evaluated separately.

My recommendation is **(c)**, with (a) as the fallback if (c) proves unstable with
only four events. This is a real design change and should be approved explicitly.

**Condition 2 — Event-level evaluation only.**
Point-level metrics over 1.5 M rows and 1.71% positive time are not interpretable
here, and point-adjust would inflate results. Evaluation must be: recall over the
4 windows with a stated detection tolerance, alarms/day, detection delay, and
event-level precision. **Statistical claims must be limited to descriptive
statistics with wide intervals — four events do not support significance testing
of AUC differences.** The README must say this.

**Condition 3 — Mandatory data masks.**
Mask runs of ≥ 1 h constant analog values (the 51 h block and its 6 siblings) and
do not impute across outages > 1 h. A mask file must be a build artifact, and a
test must assert that masked rows never enter calibration.

### What would change the verdict to NO-GO

- If Condition 1 cannot be satisfied without conditioning on the fault channel
  itself, the "context-aware" claim collapses into "detect the state change",
  which is a much weaker and more generic project. In that case the honest move is
  to switch the primary dataset to NAB, where context is calendar-only and cannot
  be circular, and keep MetroPT-3 as a secondary stress test.
- If the frozen block or outages turn out to overlap the labelled windows after
  closer inspection, label quality would be insufficient for event timing.

Neither has happened yet.

---

## 8. Day 0 gate checklist

| Requirement | Status |
| --- | --- |
| Dataset downloaded, extracted, checksummed | done |
| Structure, dimensions, dtypes verified directly | done |
| Timestamps, ranges, duplicates, missingness verified | done |
| Sampling frequency and irregularity quantified | done |
| Sensor distributions and operating-state encoding verified | done |
| Failure labels and durations verified | done |
| Label suitability assessed | done — event-level only |
| Context sparsity quantified | done — back-off required |
| Figures generated | 10 figures, all from script |
| `docs/datasets.md` | done |
| `docs/day0_report.md` | this file |
| `scripts/download_data.py` | done |
| `tests/test_data_schema.py` | done, 10 passed |
| No dataset committed to Git | `.gitignore` covers `data/` |
| No models, detectors, CCCS, or app code | confirmed — none written |

**Awaiting explicit approval before Day 1.**
