# Day 2 — Data layer, masks, chronology, leakage

**Date:** 2026-11-16
**Scope:** deterministic loading, schema contract, 1-minute grid, quality masks
M1–M3, hysteresis state, lagged context, chronological folds, leakage tests.
**Out of scope and not performed:** model training, Isolation Forest, Score A/B,
CCCS calibration, alarm-budget solving, Streamlit, any Day 3 work.

**Verdict:** **CONDITIONAL GO** (conditions in §11). The Day 2 gates pass, but two
measured data properties must be handled explicitly at the start of Day 3 before
any score is fitted.

---

## 1. What was built

A real package now exists at `src/anomaliradar/`. The data layer has a single
public orchestration point:

```python
from anomaliradar.data.loader import load_native
from anomaliradar.data.pipeline import build_dataset

ds     = build_dataset(load_native())   # masks, state, context
folds  = ds.folds()                     # 1,2,3,4,T
split  = ds.split(folds[0])             # positional indices, leakage-checked
X      = ds.feature_matrix()            # causal context only
```

Module responsibilities:

| Module | Responsibility |
| --- | --- |
| `timewindow.py` | half-open `TimeWindow` `[start, end)`, used by masks and splits |
| `data/schema.py` | column contract, numeric/binary/range checks, R1/R4 channel partition |
| `data/loader.py` | deterministic CSV read, alias normalisation, timestamp parse, sort, validate |
| `data/labels.py` | four verified events, half-open representation, detection credit `τ = min(6 h, |E|/4)` |
| `data/grid.py` | 1-min aggregation, ≤2-min forward-fill, break/segment ids |
| `data/masking.py` | M1 frozen blocks, M2 major gaps, M3 warm-up, per-event summary |
| `data/state.py` | raw state, 3-minute hysteresis, run length, transitions |
| `data/features.py` | lagged state/run-length/transitions, day type, hour; causal-column guard |
| `data/splits.py` | fold construction, L1/L2 validation, split materialisation |
| `data/pipeline.py` | `Dataset`, `build_dataset`, `context_ready`, summary/fingerprint |

No model, score, calibration or alarm object exists in the codebase.

---

## 2. Deterministic loading and schema validation

Verified against the raw payload (`data/raw/metropt3/MetroPT3(AirCompressor).csv`):

- **1,516,948 native rows**, 16 columns (`timestamp` + 7 analog + 8 digital).
- Timestamps parse and sort monotonically; **zero duplicates**; native rows are
  complete (no missing values).
- Column names are the payload names (`DV_eletric`, `Towers`, `Caudal_impulses`),
  with explicit aliases for the UCI page/PDF spellings.
- The unnamed decisecond counter is dropped.
- Range checks reject unit/parse errors (`TP*`, `H1`, `DV_pressure`, `Reservoirs`
  in `[-1, 15]` bar; `Oil_temperature` in `[-40, 120]`; `Motor_current` in `[-1, 30]`).
- Digital channels are asserted binary.

Loading the same file twice produces byte-identical frames; `Dataset.fingerprint()`
is content-stable across processes (object columns are hashed through pandas, not
via pointer bytes).

---

## 3. 1-minute grid, gaps and forward-fill

Native sampling is 10 s with jitter. The grid spans every minute between the first
and last observation:

| Quantity | Value |
| --- | --- |
| Grid minutes | **306,960** (5,116 h = 213.2 days) |
| Minutes with ≥1 native sample | **252,720** |
| Minutes with a value after fill | **252,731** (11 forward-filled) |
| Segments | **321** |
| Breaks > 2 min | **320** |

Rules as implemented:

- Analog channels use the mean of samples in the minute; digital channels use the
  **last** sample (averaging a switch would invent a value).
- A missing minute is forward-filled **only** when the bracket to the surrounding
  observations is ≤ 2 min. Forward-fill copies the last value; it never
  interpolates and never writes zero.
- Any larger gap is a break: no fill, no interpolation, and `segment_id` increments
  so state, run length and lag reset.
- Grid rows are flagged `observed`, `ffilled`, `has_value`, so no downstream code
  can mistake a fill for an observation.

---

## 4. Masks M1–M3

All masks are rule-based and **label-free**. They are a pure function of the sensor
frame, so they are computed once and applied unchanged to every fold. They are
retrospective quality filters, not causal live features (see risk R5).

### 4.1 Global mask budget

| Mask | Rule | Result |
| --- | --- | --- |
| **M1 frozen** | all 7 analog channels constant for ≥ 360 consecutive native rows | **7 blocks**, 50,425 native rows = **3.3241%**, max **51.4306 h**; 10,155 grid rows (3.3082%) |
| **M2 major gap** | native interval > 1 h | **160 gaps**, **868.513 h**; 52,108 grid rows |
| **M3 warm-up** | first 30 min after each M1 block or M2 gap | **5,010** rows (210 after frozen + 4,800 after gaps) |
| **Combined** | M1 ∪ M2 ∪ M3 | **66,778** rows = **21.7546%** of wall clock |
| **Usable observed** | observed ∧ ¬combined | **238,179** rows = **77.593%** of wall clock (**94.25%** of observed rows) |

The large M2 share is the machine's duty schedule, not a defect. The 10% stop
condition applies **per event window**, not globally.

### 4.2 Overlap with failure windows (M1 and M2 reported separately)

| Event | Window | M1 | M2 | M3 | Combined | Share |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.00 h (1,440 min) | **19 min** (1.3194%) | **0** | **30 min** (2.0833%) | **49 min** | **3.4028%** |
| 2 | 6.52 h (391 min) | **0** | **0** | **0** | **0** | **0.0000%** |
| 3 | 52.52 h (3,151 min) | **0** | **238 min** (7.5532%) | **30 min** (0.9521%) | **268 min** | **8.5052%** |
| 4 | 4.52 h (271 min) | **0** | **0** | **0** | **0** | **0.0000%** |

**Event 3 re-verification (required stop check):** combined mask **268 / 3,151 =
8.5052%**, below the 10% limit with a **1.49-point margin**. PASS.

Two corrections to earlier days, both documented in
`docs/methodology.md` §12 and `docs/experimental_protocol.md` §11:

1. **Event 1 M1 = 19 min.** Day 0 computed a frozen block's end as
   `start + rows × 10 s`. The block `2020-04-17 09:20:33 → 2020-04-18 00:18:07`
   contains internal gaps, so its real timestamp end is 00:18:07, not the
   arithmetic 21:45:33. The corrected tail crosses midnight into event 1, followed
   by a 30-minute M3. This is a timestamp-arithmetic fix, not a new mask.
2. **Event 3 M2 = 238 min, not 3.78 h.** Two gaps touch the window: the fully
   inside 3.784 h gap (06-06 15:55:17 → 19:42:19) and the first 0.189 h of the
   21.47 h gap starting 06-07 14:19:39, just before the window closes. Day 1
   counted only the first, reporting 7.2%; the strict value is **7.5532%**.

**Detection-credit coverage is a separate question from full-window coverage.**
The credit window is `[t_start − 2 h, t_start + τ)`:

| Event | Credit window minutes | Masked | Usable | Usable share |
| --- | --- | --- | --- | --- |
| 1 | 480 | **169** | 311 | **64.79%** |
| 2 | 218 | 0 | 218 | 100% |
| 3 | 480 | 0 | 480 | 100% |
| 4 | 188 | 0 | 188 | 100% |

Event 1 is the only event whose **credit** window is degraded: the frozen block
runs from 04-17 22:00 (the credit start) to 04-18 00:18, plus warm-up to 00:48, so
the first usable credit row is **2020-04-18 00:49**, 49 minutes after the event
start. Detection is still possible for the remaining 5 h 11 min, but no alarm can
be credited before 00:49. Event 3's 8.51% full-window cost falls entirely **after**
its credit window (06-05 08:00–16:00), so detection credit there is unobstructed.

---

## 5. Operating state and lagged context

Raw state (`Motor_current`, `DV_eletric`): `off` / `offloaded` / `loaded` /
`other`, with `start` absorbed into `loaded`. A candidate must persist **3
consecutive grid minutes** before it is confirmed; this 3-minute hysteresis is a
data-layer state definition and is **not** the 30-minute alarm debounce.

Confirmed state on the grid:

| State | Rows | Share | Episodes | Median run |
| --- | --- | --- | --- | --- |
| `off` | 153,465 | 49.9951% | 9,805 | 14 min |
| `offloaded` | 80,322 | 26.1669% | 10,071 | 7 min |
| `loaded` | 16,932 | 5.5160% | 700 | 4 min |
| `other` | 2,012 | 0.6555% | 175 | 4 min |

Context features (all strictly causal):

- `state_lag1` — confirmed state at `t−1`, **within the same segment**; NaN at a
  segment start.
- `run_length_lag1_min` — lagged confirmed run length.
- `transitions_1h_lag1` — transitions in the 60 rows strictly before `t`.
- `day_type`, `hour` — calendar, known in advance.

`FEATURE_COLUMNS` excludes `state`, `raw_state`, `run_length_min`,
`is_transition`, `is_other_state`; a guard raises if any same-row state column is
proposed as a feature. The context cell is `(s_{t-1}, day_type)`.

Feature NaN shares: `state_lag1` 17.67%, `run_length_lag1_min` 17.67%,
`transitions_1h_lag1` 0.105%, `day_type`/`hour` 0%.

Rows whose **lagged** state is `other` are excluded from calibration (not only
rows whose current state is `other`), so the realized calibration cells are
exactly the six declared ones.

---

## 6. Chronological splits

Every event gets its own rolling-origin fold. Test windows start 12 h before the
label for context and score traces; detection credit remains 2 h before the label.

| Fold | Event | Train | Calibration | Embargo | Test window |
| --- | --- | --- | --- | --- | --- |
| 1 | #1 | 02-01 00:00 → 03-21 00:00 | 03-21 00:00 → 04-11 00:00 | 6.5 d | 04-17 12:00 → 04-20 00:00 |
| 2 | #2 | 02-01 00:00 → 05-01 23:30 | 05-01 23:30 → 05-22 23:30 | 6.5 d | 05-29 11:30 → 05-31 06:01 |
| 3 | #3 | 02-01 00:00 → 05-08 10:00 | 05-08 10:00 → 05-29 10:00 | 6.5 d | 06-04 22:00 → 06-08 14:31 |
| 4 | #4 | 02-01 00:00 → 06-17 14:30 | 06-17 14:30 → 07-08 14:30 | 6.5 d | 07-15 02:30 → 07-16 19:01 |
| **T** | false-alarm control | fold-4 train | fold-4 calibration | 23.4 d | 08-01 00:00 → 09-01 04:00 |

Detection-credit windows: fold 1 `04-17 22:00 → 04-18 06:00`; fold 2
`05-29 21:30 → 05-30 01:07:45`; fold 3 `06-05 08:00 → 06-05 16:00`; fold 4
`07-15 12:30 → 07-15 15:37:45`.

Materialised row counts:

| Fold | Train | Calibration | Test (kept) | Test window | Detection | Labels excluded from train |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 58,620 | 25,340 | 2,807 | 3,600 | 311 | 0 |
| 2 | 101,664 | 24,550 | 2,280 | 2,551 | 218 | 1,440 (event 1) |
| 3 | 109,970 | 20,501 | 3,652 | 5,311 | 480 | 1,440 (event 1) |
| 4 | 150,307 | 20,488 | 2,415 | 2,431 | 188 | 4,982 (events 1–3) |
| T | 150,307 | 20,488 | 35,940 | 44,880 | 0 | 4,982 |

Calibration label exclusion is **0** in every fold (calibration never overlaps a
labelled window by construction). Train folds 2–4 remove earlier events; fold 4's
4,982 excluded rows are the sum of events 1–3 falling before its train cut.

### 6.1 Calibration-cell density (Day 3 input)

Counts of `(lagged state, day type)` rows in calibration:

| Fold | off wd | off we | offloaded wd | offloaded we | loaded wd | loaded we | Minimum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12,481 | 4,132 | 4,942 | 1,666 | **701** | 1,418 | 701 |
| 2 | 11,100 | 3,981 | 5,320 | 1,891 | 2,206 | **52** | 52 |
| 3 | 8,501 | 2,769 | 5,315 | 1,656 | 2,168 | **92** | 92 |
| 4 | 8,370 | 3,605 | 6,087 | 2,349 | **48** | **29** | 29 |
| T | 8,370 | 3,605 | 6,087 | 2,349 | **48** | **29** | 29 |

All folds contain exactly the six declared cells; `other` never appears after the
lagged filter.

---

## 7. Leakage results

| Rule | What was checked | Result |
| --- | --- | --- |
| **L1** | `max(train) < min(calib) ≤ max(calib) < min(test)` per fold | **PASS**, all folds |
| **L2** | calibration ends 7 days before the event **and** raw buffer to test start ≥ 6 days | **PASS**; 6.5 d raw buffer, 23.4 d for fold T (reinterpretation recorded in the protocol changelog) |
| **L3** | poison all rows after `t`; features at `t` unchanged | **PASS** (`test_poisoning_the_future_does_not_change_past_features`) |
| **L4** | masks are a pure function of the sensor frame, not of labels | **PASS**: mask signature and fingerprint identical under two different label sets |
| **L5** | masked rows in no fit set | **PASS**: train, calibration, test windows checked |
| **L6** | labels never enter fitting or calibration | **PASS**: `build_dataset` has no label argument; train/calibration contain zero labelled rows |
| **L7** | context is `s_{t−1}`, never `s_t` | **PASS**: row-by-row shift check; same-row state columns rejected by guard |
| **L9** | gaps reset state; no statistic spans a > 1 h gap | **PASS**: constructed-gap fixture; run length/lag reset |
| **L8, L10** | alarm α depends only on calibration; scaler fitted on train | **Deferred to Day 3** — no model or scaler exists yet, so there is nothing to test |

Test rows retain their labels (they are the evaluation). Detection indices are a
subset of test indices in every fold.

---

## 8. Determinism and reproducibility

- `scripts/day2_data_audit.py` recomputes every number in this report and writes
  `reports/day2/data_audit.json` plus four CSV tables.
- Two consecutive runs produce byte-identical JSON and CSV output (verified with
  `diff`).
- Dataset fingerprint: `420986c98d243c84…`.
- No randomness, no sampling, no environment-dependent ordering.

Reproduction:

```bash
.venv/bin/python scripts/day2_data_audit.py
.venv/bin/python -m pytest tests/ -q
```

---

## 9. Test suite

**106 passed in ~21 s** with the real CSV present.

| File | Tests | Focus |
| --- | --- | --- |
| `test_data_schema.py` | 12 | contract, aliases, duplicates, binary, ranges |
| `test_grid.py` | 10 | aggregation, fill limit, breaks, segment resets |
| `test_state.py` | 14 | raw mapping, 3-min hysteresis, run length, transitions |
| `test_masking.py` | 15 | M1/M2/M3 thresholds, warm-up, label independence, reporting |
| `test_labels.py` | 5 | four events, inclusive ends, τ cap, tail window |
| `test_features.py` | 10 | lag, run length, transitions, calendar, R1 partition |
| `test_splits.py` | 14 | fold boundaries, ordering, disjointness, validation |
| `test_leakage.py` | 9 | L3/L4/L5/L6/L7/L9, long-gap behaviour |
| `test_real_data.py` | 17 | real stop conditions, masks, splits, calibration cells (skips if CSV absent) |

The synthetic suite is self-contained: **89 tests pass in 10.7 s** with
`tests/test_real_data.py` excluded. The 17 real-data tests `pytest.skip` when the
CSV is missing; no test requires the download.

`pyproject.toml` sets `pythonpath = ["src"]` and filters the upstream
pandas/numpy timedelta `DeprecationWarning`.

---

## 10. Files created / changed

**Created**

- `pyproject.toml`
- `src/anomaliradar/__init__.py`, `src/anomaliradar/timewindow.py`
- `src/anomaliradar/data/{__init__,schema,loader,labels,grid,masking,state,features,splits,pipeline}.py`
- `tests/{conftest,helpers,test_data_schema,test_grid,test_state,test_masking,test_labels,test_features,test_splits,test_leakage,test_real_data}.py`
- `scripts/day2_data_audit.py`
- `reports/day2/{data_audit.json,masks_by_event.csv,folds.csv,splits.csv,calibration_cells.csv}`
- `docs/day2_report.md` (this file)

**Changed**

- `docs/experimental_protocol.md` — L2 reinterpreted (7-day event margin + ≥6-day
  raw buffer), L4 clarified (mask is a pure function of the sensor frame),
  changelog entries.
- `docs/methodology.md` — corrected M1 block end/event-1 overlap, event-3 two-gap
  M2 value, verified mask table, lagged-`other` calibration exclusion, confirmed
  state shares and run lengths, per-fold calibration-cell density, Day 2 revision
  notes.

---

## 11. Unresolved risks and limitations

| ID | Risk | Evidence | Required Day 3 action |
| --- | --- | --- | --- |
| **R1** | **Fold 4 `loaded` calibration is thin.** | 48 weekday + 29 weekend = **77 rows**; both fine cells are below the 50-row refusal floor, so calibration must back off to the pooled state cell (77), still below the 200-row density target. | Pre-register the back-off path and label fold-4 `loaded` calibration as low-confidence. Do not silently pool across day types. |
| **R2** | **Event 1 detection credit is 35% masked.** | 169 / 480 credit minutes masked; first usable credit row `04-18 00:49`, 49 min after onset. | Evaluate event 1 detection delay from 00:49 and report the 2 h 49 min blind interval explicitly; do not count it as a miss. |
| **R3** | **Fold 2 `loaded∣weekend` is only 52 rows.** | Just above the 50-row refusal floor, far below 200. | Same back-off policy as R1; report cell-level confidence. |
| **R4** | **Masks are retrospective, not causal.** | M1 can only be declared after a frozen block ends; M2 after a gap closes. | Do not use M1/M2/M3 as live features. For streaming, apply masks with the unavoidable detection latency or use a separate causal quality gate. |
| **R5** | **17.67% of rows have no lagged state** (segment starts). | `state_lag1` NaN share. | Define Score B behaviour for context-free rows (likely no B contribution). Score A can still run. |
| **R6** | **Hysteresis lag.** | Up to `min_dwell − 1 = 2` minutes. | Accept and document; negligible against 4.5–52 h events. |
| **R7** | **Grid aggregation smooths short transients.** | 1-min mean. | Accepted: all labelled events last ≥ 4.5 h. |
| **R8** | **Fold T is a single 31-day false-alarm period.** | 35,940 test rows, no event. | Report a single budget result without confidence intervals, or add a rolling sub-window estimate. |
| **R9** | **Static `other` state is not calibratable.** | 2,012 confirmed rows (0.66%); lagged `other` excluded. | Score B must back off or abstain for `other`; do not create a seventh cell. |

No unresolved **Day 2** failure exists. R1–R3 are calibration-robustness risks that
belong to Day 3; R4–R9 are documented limitations of the data layer.

---

## 12. Verdict

**CONDITIONAL GO.**

Day 2 deliverables are complete and the gates pass:

- deterministic load and schema validation on 1,516,948 native rows;
- 306,960-minute grid with explicit missingness and break semantics;
- M1/M2/M3 masks with M1 and M2 reported separately per event;
- **event 3 re-verified at 8.5052% < 10%** (M1 0, M2 7.5532%, M3 0.9521%);
- 3-minute state hysteresis, lagged context, exact six calibration cells;
- five chronological folds plus tail, all L1/L2 checks passing;
- leakage invariants L3–L7 and L9 verified by tests;
- **106 tests green** (89 synthetic-only tests green without the real CSV);
- reproducible audit artefacts.

The verdict is conditional on three pre-registered Day 3 actions, taken **before**
any score is computed:

1. Fold 4's `loaded` calibration must use the documented state-level back-off and
   be reported as low-confidence (**77 rows**).
2. Event 1 detection credit must be evaluated from the first usable row
   (`2020-04-18 00:49`) and its 169 masked credit minutes reported explicitly.
3. Masks must not be treated as causal live features; any streaming use must
   account for M1/M2 detection latency.

Day 2 stops here. No model, score, calibration, alarm algorithm or UI was built.
Awaiting approval before Day 3.
