# Day 3 — Scores, calibration, alarm budget, baselines and evaluation

**Date:** 2026-11-16
**Dataset:** MetroPT-3 (real), `data/raw/metropt3/MetroPT3(AirCompressor).csv`
**Config hash:** `06954d34ced38410`
**Dataset fingerprint:** `420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed`
**Run command:** `.venv/bin/python scripts/run_experiments.py --output reports/experiments`
**Figure command:** `.venv/bin/python scripts/make_figures.py`

**Verdict:** **CONDITIONAL GO** (conditions in §10). All nine pre-registered
falsification criteria F1–F9 did **not** trigger on the real data. The two-score
union detects **2 of 4** labelled events at the primary 1/day budget, which is
the best of the pre-registered primary methods on recall and false-alarm count,
but the budget response is coarse,
fold 4's loaded-state calibration is thin, and 4 events / 7.75 observed event-fold
days cannot support a tight false-alarm claim.

---

## 1. Executive summary

| Question | Answer |
| --- | --- |
| Did all tests pass? | **Yes.** 141 passed in 87.34 s (106 Day 0–2 + 35 Day 3). |
| Did the real experiment run reproducibly? | **Yes.** A second full run is byte-identical on fold results, alarms, summary, falsification and all four prediction files. |
| Best primary method at 1/day | **OR (`P3_or`)**: 2/4 events, 11 alarms, 1 false alarm on the four event folds. |
| Primary Score A | `P1_A_hybrid`: 1/4 (event 3, 10 min early), 2 alarms, 0 false. |
| Primary Score B | `P2_B_B1`: 1/4 (event 4, 32 min late), 9 alarms, 1 false. |
| Tail (31 wall days, 24.96 observed) | OR 0.5609 alarms/day (requested 1.0, −43.9%); all 14 tail alarms false. |
| F1–F9 | **9 PASS, 0 FAIL.** Several passes are narrow or low-power and are flagged in §5. |
| Can a tight 1/day claim be made? | **No.** See the budget-coarseness evidence in §4.2 and §4.4. |

The headline is honest rather than flattering: one event is found by value
anomalies only (event 3), one by state persistence only (event 4), and two
(event 1, event 2) are missed by both primary scores and by the union. Event 1
is also the event whose *detection-credit* window is 35.2% masked, so even a
correct alarm there could only be credited on 311 of 480 minutes.

---

## 2. What was implemented

| Module | Responsibility |
| --- | --- |
| `src/anomaliradar/model/config.py` | frozen config, hash, budget grid, channels, arms, variants |
| `src/anomaliradar/model/common.py` | segment-aware shifts, rolling mean/MAD, EWMA, train-only standardization |
| `src/anomaliradar/model/score_a.py` | 5-channel Ridge, four context arms, causal nonconformity |
| `src/anomaliradar/model/score_b.py` | confirmed-state runs, survival reference, B1/B3, leave-one-run-out |
| `src/anomaliradar/model/calibration.py` | chronological reference/selection split, Mondrian cells, back-off, refusal |
| `src/anomaliradar/model/alarm.py` | 30-min debounce, budget solver, eligibility enforcement |
| `src/anomaliradar/model/fusion.py` | A, B, OR, AND, weighted, Bonferroni reporting column |
| `src/anomaliradar/model/baselines.py` | global robust z, rolling robust z, IForest raw/context |
| `src/anomaliradar/model/evaluation.py` | event recall, delays, alarm volume, per-cell FPR, exposure |
| `src/anomaliradar/model/runner.py` | folds, ablations, tail reuse, metrics, F1–F9 |
| `scripts/run_experiments.py` | deterministic CLI |
| `scripts/make_figures.py` | deterministic figures from frozen CSVs |

Design details that matter for reading the numbers:

- **Score A** uses only `Reservoirs`, `TP3`, `H1`, `DV_pressure`,
  `Oil_temperature`; TP2 is excluded. Features are strictly causal (lags 1/5,
  rolling means 15/60/360 min, rolling MAD 60 min, lagged context only). Ridge
  `alpha=1.0` per channel. Nonconformity is
  `log1p(|e| / max(causal EWMA scale, train median |e|))`, then
  `a_t = max_c a_{c,t}`.
- **Score B** is a function of confirmed state and run length only. It never reads
  a value channel. One run cannot bridge a data gap; run boundaries reset at
  segments. The reference is state-specific if at least 50 completed runs exist,
  otherwise pooled; the conformal reference rows are restricted to completed
  reference runs (C10).
- **Calibration** is empirical, not distribution-free. The earliest 2/3 of
  eligible calibration rows build the reference; the latest 1/3 selects `alpha`
  (C1). Cells are `(lagged state, day type)`, then state, then global. Levels
  need `n >= 200`; `50 <= n < 200` is used and flagged `*_low`; `n < 50` refuses
  (`p=1`, `eligible=False`). Every scored row records level and `n`.
- **Alarm layer** has a 30-minute debounce, distinct from the 3-minute state
  hysteresis. The budget solver sees calibration data only. The tail reuses the
  frozen fold-4 model and never refits.
- **Fusion** is compared under the identical alarm layer: A, B, OR
  (`min(pA,pB)`), AND (`max`), weighted `λ=0.5` primary with the full grid
  reported. OR also has a Bonferroni column (`min(1,2·min)`) for reporting only.

---

## 3. Experiment as executed

### 3.1 Folds

| Fold | Event | Train rows | Calibration rows | Test rows | Detection-credit rows |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 58,620 | 25,340 | 2,807 | 311 |
| 2 | 2 | 101,664 | 24,550 | 2,280 | 218 |
| 3 | 3 | 109,970 | 20,501 | 3,652 | 480 |
| 4 | 4 | 150,307 | 20,488 | 2,415 | 188 |
| T | none (tail) | 150,307 | 20,488 | 35,940 | 0 |

Event folds contain only **7.746 observed days** in total; the tail contains
**24.958 observed days** of a 31-day wall-clock window. The event-fold sample is
small, and every rate below is a count divided by observed minutes, not calendar
minutes.

### 3.2 Calibration reference / selection sizes

| Fold | Reference rows | Selection rows | Refused selection rows |
| --- | ---: | ---: | ---: |
| 1 | 16,893 | 8,447 | 0 |
| 2 | 16,366 | 8,184 | 0 |
| 3 | 13,667 | 6,834 | 0 |
| 4 | 13,658 | 6,830 | 0 |

No selection row was refused on these folds. That does **not** mean every test
row was calibratable: Score B refused 9 test rows in fold 1 and 10 in fold 3
(short loaded cells). Those rows were scored as `p=1` and could not alarm.

### 3.3 Budgets and primary settings

Budgets `{0.5, 1.0, 2.0, 4.0}` alarms per observed day. Primary context arm
`hybrid`. Primary Score B `B1`. Primary fusion OR. Primary budget **1.0/day**.
No threshold, cell definition, arm or λ was chosen from test results. C1–C10
in `docs/experimental_protocol.md` were written before the full run.

---

## 4. Real results

### 4.1 Headline at 1/day (four event folds)

Rates marked "pooled" are total alarms divided by total observed days
(7.746). "Mean error" is the mean over folds of |realized − requested| / requested.

| Method | Recall | Events | Alarms | False | Alarms/day (pooled) | False/day (pooled) | Mean budget error |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `P1_A_hybrid` | 1/4 | 3 | 2 | 0 | 0.258 | 0.000 | 0.803 |
| `P2_B_B1` | 1/4 | 4 | 9 | 1 | 1.162 | 0.129 | 1.842 |
| **`P3_or`** | **2/4** | **3, 4** | **11** | **1** | **1.420** | **0.129** | **1.644** |
| `P3_weighted` (λ=0.5) | 0/4 | — | 5 | 3 | 0.646 | 0.387 | 0.704 |
| `P3_and` | 0/4 | — | 2 | 1 | 0.258 | 0.129 | 0.798 |
| `B1_global_z` | 2/4 | 2, 3 | 9 | 7 | 1.162 | 0.904 | 1.039 |
| `B2_rolling_z` | 0/4 | — | 7 | 7 | 0.904 | 0.904 | 1.543 |
| `B3_iforest_raw` | 1/4 | 2 | 12 | 6 | 1.549 | 0.775 | 1.419 |
| `B4_iforest_context` | 1/4 | 4 | 22 | 10 | 2.840 | 1.291 | 3.780 |

The union matches the best primary single-score recall (2/4) while keeping the
lowest false-alarm count among the 2/4 methods (1 vs 7 for global z). It does so
at a higher mean budget error than A or weighted fusion, and with a pooled rate
of 1.42/day for a 1/day request.

Per-fold union outcomes:

| Fold | Event | A | B | OR | First alarm delay | Alarms (OR) |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | 1 | miss | miss | miss | — | 0 |
| 2 | 2 | miss | miss | miss | — | 0 |
| 3 | 3 | **hit** | miss | **hit** | **−10 min** | 2 |
| 4 | 4 | miss | **hit** | **hit** | **+39 min** | 9 |

B's first alarm for event 4 is at **+32 min** (`p_b=4.48e-4`), below B's own
`α=5.69e-4` but above the union's stricter `α=3.26e-4`. The union's first accepted
alarm is therefore at **+39 min**, when `p_b` falls to its floor `7.5e-5`. The
39-minute union delay is caused by the stricter union threshold, not by the
debounce. The fold-4 union emits 9 alarms at 1/day (5.37/day): 1 is a cell-level
`offloaded` false alarm at 09:09, and 8 are global-back-off `loaded` alarms
during the event. The 8 event alarms are a run-length flood at Score B's p-value
floor, not a calibrated 1/day process.

### 4.2 Budget sweep on the four event folds

Recall is events detected out of 4. "Mean rate" is the mean of the per-fold
alarms/day; it gives each fold equal weight and therefore differs from the pooled
rate in §4.1.

| Method | Budget | Recall | Alarms | False | Mean rate | Mean abs budget error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A hybrid | 0.5 | 0/4 | 1 | 0 | 0.099 | 0.803 |
| A hybrid | 1.0 | 1/4 | 2 | 0 | 0.197 | 0.803 |
| A hybrid | 2.0 | 1/4 | 12 | 9 | 1.504 | 0.357 |
| A hybrid | 4.0 | 1/4 | 21 | 17 | 2.792 | 0.473 |
| B1 | 0.5 | 0/4 | 0 | 0 | 0.000 | 1.000 |
| B1 | 1.0 | 1/4 | 9 | 1 | 1.342 | 1.842 |
| B1 | 2.0 | 1/4 | 9 | 1 | 1.342 | 1.171 |
| B1 | 4.0 | 3/4 | 150 | 5 | 16.677 | 3.590 |
| **OR** | 0.5 | 1/4 | 10 | 1 | 1.440 | 2.986 |
| **OR** | 1.0 | **2/4** | 11 | 1 | 1.539 | 1.644 |
| **OR** | 2.0 | 2/4 | 13 | 3 | 1.736 | 0.974 |
| **OR** | 4.0 | 2/4 | 24 | 13 | 3.301 | 0.793 |
| Weighted | 0.5 | 0/4 | 2 | 1 | 0.298 | 1.096 |
| Weighted | 1.0 | 0/4 | 5 | 3 | 0.704 | 0.704 |
| Weighted | 2.0 | 0/4 | 17 | 11 | 2.320 | 0.980 |
| Weighted | 4.0 | 1/4 | 22 | 13 | 3.003 | 0.718 |
| AND | 0.5 | 0/4 | 2 | 1 | 0.298 | 1.096 |
| AND | 1.0 | 0/4 | 2 | 1 | 0.298 | 0.798 |
| AND | 2.0 | 0/4 | 15 | 12 | 2.174 | 1.202 |
| AND | 4.0 | 0/4 | 19 | 13 | 2.627 | 0.738 |

Two properties are important and unflattering:

1. **Score B's realized rate is a step function.** 1/day and 2/day both produce
   9 event-fold alarms and 1.342/day; 4/day jumps to 150 alarms (16.7/day).
   0.5/day produces zero. The discrete survival p-value grid cannot express
   intermediate rates.
2. **OR inherits that coarseness.** 0.5/day already produces 10 alarms
   (1.44/day) because the union's own α is selected on the min-p statistic; its
   rate is not the sum of A's and B's rates. 1/day adds one alarm. The union's
   response between 0.5 and 2/day is nearly flat.

Per-cell FPR deviation also grows sharply above 1/day. At 1/day the largest
per-cell deviation is 1.64% (OR fold 4). At 2–4/day, a one-row `NaN` cell in
fold 3 gives empirical FPR 1.0, and fold 1's `loaded|weekend` cell has 3 alarms
in 10 test rows (FPR 0.30). These are small-cell artifacts, not evidence of a
well-calibrated cell; they are the reason the 1/day budget is the only one with
clean per-cell support.

### 4.3 Detection delays

Only detected events appear. Negative values are early alarms.

| Method | Event 1 | Event 2 | Event 3 | Event 4 |
| --- | ---: | ---: | ---: | ---: |
| A hybrid | — | — | **−10** | — |
| B1 | — | — | — | **+32** |
| OR | — | — | **−10** | **+39** |
| Global z | — | **−14** | **−10** | — |
| Rolling z | — | — | — | — |
| IForest raw | — | **−13** | — | — |
| IForest + context | — | — | — | **−101** |

Event 3's early alarm is 10 minutes before the labelled start but inside the
detection-credit window, which begins 8 hours before the event. Event 4's union
delay is 39 minutes, not the 32 minutes of the underlying B alarm, because the
union threshold is stricter than B's standalone threshold.

### 4.4 Tail: frozen fold-4 model, 31 wall days (24.96 observed)

The tail has no labelled event, so every tail alarm is false by construction.
This table is a false-alarm control test, not a detection test.

| Method | Budget | Alarms | Realized/day | Relative error | Reachable | Coarse |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A hybrid | 0.5 | 9 | 0.3606 | −27.9% | yes | no |
| A hybrid | 1.0 | 38 | 1.5225 | **+52.3%** | yes | no |
| A hybrid | 2.0 | 43 | 1.7229 | −13.9% | yes | no |
| A hybrid | 4.0 | 124 | 4.9683 | +24.2% | yes | no |
| B1 | 0.5 | 0 | 0.0000 | −100% | yes | no |
| B1 | 1.0 | 35 | 1.4023 | +40.2% | yes | no |
| B1 | 2.0 | 35 | 1.4023 | −29.9% | yes | no |
| B1 | 4.0 | 113 | 4.5275 | +13.2% | yes | no |
| **OR** | 0.5 | 14 | 0.5609 | +12.2% | yes | no |
| **OR** | 1.0 | **14** | **0.5609** | **−43.9%** | yes | no |
| **OR** | 2.0 | 58 | 2.3239 | +16.2% | yes | no |
| **OR** | 4.0 | 101 | 4.0467 | +1.2% | yes | no |
| Weighted | 0.5 | 19 | 0.7613 | +52.3% | yes | no |
| Weighted | 1.0 | 37 | 1.4825 | +48.2% | yes | no |
| Weighted | 2.0 | 104 | 4.1669 | +108.3% | yes | no |
| Weighted | 4.0 | 167 | 6.6912 | +67.3% | yes | no |
| AND | 0.5 | 17 | 0.6811 | +36.2% | yes | no |
| AND | 1.0 | 38 | 1.5225 | +52.3% | yes | no |
| AND | 2.0 | 80 | 3.2053 | +60.3% | yes | no |
| AND | 4.0 | 151 | 6.0501 | +51.3% | yes | no |
| Global z | 1.0 | 17 | 0.6811 | −31.9% | yes | no |
| Rolling z | 1.0 | 62 | 2.4841 | +148.4% | yes | no |
| IForest raw | 1.0 | 6 | 0.2404 | −76.0% | yes | no |
| IForest + context | 1.0 | 11 | 0.4407 | −55.9% | yes | no |

F4 is evaluated on the primary OR, whose 1/day error is −43.9%, just inside the
50% limit. B (+40.2%) and weighted (+48.2%) are also inside the limit; A and AND
(+52.3%) exceed it. The pass is a 6.1-point margin for OR and is method-specific.
The 0.5 and 1/day OR rows are the same 14 alarms: again a step function, not
proportional control.

### 4.5 Calibration audit

- **Reference/selection separation:** chronological, disjoint, no labels, no
  test rows. Sizes in §3.2.
- **Back-off frequency at 1/day (all emitted alarms):**

  | Method | Alarms | Cell-level | State-level | Global | `n < 200` | `n < 50` | Refused rows |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | A hybrid | 40 | 38 | 0 | 2 | 0 | 0 | 0 |
  | B1 | 44 | 36 | 0 | 8 | 0 | 0 | 19 test rows |
  | OR | 25 | 15 | 0 | 10 | 0 | 0 | 0 |

  (These counts include the tail; event-fold-only OR is 11 alarms, fold 3 cell 1
  + global 1, fold 4 cell 1 + global 8.)
- **Fold 4 loaded-state support:** A's loaded cell is 19 + 22 rows and its loaded
  state is 41; B's loaded cell is 16 + 14 and its loaded state is 30. All are
  below `n_refuse=50`, so every fold-4 loaded row backs off to global
  (`n=13,382`). B's event-4 detection is therefore a **global** calibration, not
  a loaded-state calibration.
- **Score B refusals:** 9 test rows in fold 1 and 10 in fold 3 are refused
  (`p=1`, `eligible=False`); they cannot alarm.
- **Budget solver:** every requested budget is reported reachable and none is
  flagged coarse, but the realized tail rate shows that "reachable" does not
  mean "proportionally controllable" for Score B. The solver never sees test or
  tail data.
- **Per-cell deviation at 1/day:** maximum 1.64% (OR fold 4), 0.081% (A fold 4),
  1.62% (B fold 4).
- **No distribution-free claim.** The series is not exchangeable; these are
  empirical calibration and empirical false-alarm-control results.

### 4.6 Context arms and suppression diagnostics

| Arm | 0.5/day | 1/day | 2/day | 4/day | False alarms at 1/day |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0/4 | 0/4 | 1/4 | 1/4 | 0 |
| calendar | 0/4 | 0/4 | 1/4 | 1/4 | 0 |
| state | 0/4 | **1/4** | 1/4 | 1/4 | 0 |
| hybrid | 0/4 | **1/4** | 1/4 | 1/4 | 0 |

The state and hybrid arms detect event 3; none and calendar do not. This is the
opposite of "state conditioning normalizes the fault": context helps here rather
than suppressing the anomaly.

Suppression is nevertheless visible in the diagnostics. The number of rows where
the no-context score would alarm at `p <= 0.05` but the context arm does not
(`suppressed_rows_p005`) is 0–1 per event window:

| Fold | Event | Hybrid mean context effect | Hybrid max context effect | Rows suppressed (`p<=0.05`) |
| --- | --- | ---: | ---: | ---: |
| 1 | 1 | 3.521 | 10.377 | 1 |
| 2 | 2 | 0.726 | 1.296 | 1 |
| 3 | 3 | 1.236 | 2.884 | 0 |
| 4 | 4 | 0.737 | 1.023 | 0 |

So context changes predictions substantially on a handful of event rows
(especially event 1) but does not erase the event-wide signal. Mean Score A
inside the detection window is slightly higher for hybrid than none in folds
1–3 (+0.0279, +0.0007, +0.0049) and slightly lower in fold 4 (−0.0084). The
mechanism is a real effect on a few rows, not a wholesale normalization.

### 4.7 Mask coverage and mask ablations

| Event | Window | Masked minutes | Masked share |
| --- | ---: | ---: | ---: |
| 1 | 1,440 min (24 h) | 49 | 3.40% |
| 2 | 391 min | 0 | 0.00% |
| 3 | 3,151 min | 268 | **8.51%** |
| 4 | 271 min | 0 | 0.00% |

Event 3 is the worst case at **8.51%**, below the 10% F6 limit by 1.49 points.
Event 1 has only 3.40% of its *event window* masked, but **169 of its 480
detection-credit minutes (35.2%) are masked**. Detection credit for event 1 can
therefore only be earned on 311 minutes. This is a real limitation on the
event-1 recall measurement and is not captured by the F6 window metric.

Mask ablations (OR, four event folds):

| Variant | Budget | Recall | Alarms | False | Realized/day | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | 1.0 | 2/4 | 11 | 1 | 1.5388 | 1.6445 |
| M-hygienic | 1.0 | 2/4 | 11 | 1 | 1.5388 | 1.6445 |
| M-none | 1.0 | 1/4 | 26 | 21 | 2.6687 | 2.8705 |
| M-none | 0.5 | 1/4 | 2 | 1 | 0.2499 | 0.5966 |

Removing masks floods fold 1 with 20 false alarms and detects event 1 only by
that flood. M-hygienic (which additionally removes long loaded runs from the
calibration window) is almost identical to primary. The masks are therefore
doing real work against a label-adjacent false-alarm mechanism, but they are not
what creates the two successful detections. The F9 comparison is
primary vs hygienic: differences 0, 0, 0, +1 events at 0.5/1/2/4 per day.

### 4.8 Baselines

At 1/day the baselines are not a strawman: global robust z also detects 2/4,
and it does so with a lower mean budget error than OR (1.039 vs 1.644). It pays
for that with **7 false alarms** (0.904/day pooled) versus OR's 1 (0.129/day),
and it detects event 2, which neither primary score finds. IForest + context
detects event 4 101 minutes early at 1/day but emits 22 alarms (2.84/day) and
has a mean budget error of 3.78. The union's advantage over the strongest naive
baseline is therefore **false-alarm volume** (1 vs 7), not recall (tied at 2/4)
and not budget error (global z is better, 1.039 vs 1.644).

### 4.9 Fusion comparison

- **OR is the only fusion rule with 2/4 at 1/day.** Global z also reaches 2/4,
  but with 7 false alarms instead of 1.
- **Weighted λ=0.5** is 0/4 at 1/day and 1/4 at 4/day. The full λ grid:

  | λ | 0.5/day | 1/day | 2/day | 4/day |
  | ---: | ---: | ---: | ---: | ---: |
  | 0.00 | 0/4 | 1/4 | 1/4 | 3/4 |
  | 0.25 | 0/4 | 0/4 | 2/4 | 2/4 |
  | 0.50 | 0/4 | 0/4 | 0/4 | 1/4 |
  | 0.75 | 0/4 | 0/4 | 0/4 | 0/4 |
  | 1.00 | 0/4 | 1/4 | 1/4 | 1/4 |

  λ=0.5 is not the best point on this grid, but it was fixed before test; the
  grid is reported as sensitivity, not used to select. λ=0 and λ=1 reduce to B
  and A respectively.
- **AND** is 0/4 at every budget. Requiring both scores to agree is too strict
  when A and B fire on different events.
- **Bonferroni OR** (`min(1,2·min)`) produces the same alarm set at 1/day on
  these folds because the correction is a monotone rescaling absorbed by the
  budget solver; it is reported for completeness, not used for significance.
- **A/B contribution at 1/day:** the 25 OR alarms across all folds and the tail
  are 11 fired by A and 14 by B. On the event folds, the 11 alarms are 2 from A
  (fold 3) and 9 from B (fold 4). The two scores are complementary, not
  redundant.

### 4.10 Score relationship

| Fold | Spearman ρ(a,b) | Top-20 Jaccard | Selection rows |
| --- | ---: | ---: | ---: |
| 1 | 0.588 | 0.000 | 8,447 |
| 2 | 0.179 | 0.000 | 8,184 |
| 3 | 0.289 | 0.000 | 6,834 |
| 4 | 0.482 | 0.000 | 6,830 |

Median ρ = 0.385, far below the 0.8 redundancy threshold. The top-20 rows of the
two scores do not overlap at all. Score B is not a re-expression of Score A.

---

## 5. F1–F9 on the real data

`PASS` means the falsification criterion did **not** hold. A pass is not proof
that the design is correct; several tests have low power on four events.

| ID | Verdict | Real evidence | Reading |
| --- | --- | --- | --- |
| **F1** | PASS | Unique events at 1/day: **3** (A only) and **4** (B only). At 0.5/day none; at 2/day 3 and 4; at 4/day 1 and 4. | The union has unique detections at the primary budget. Strongest result in the study. |
| **F2** | PASS | Gain over the best single score: +1 at 0.5/day, +1 at 1/day, +1 at 2/day, **−1 at 4/day**. Gain at 3 of 4 budgets. | Passes the pre-registered "≥2 budgets" rule. At 4/day, B alone (3/4) beats OR (2/4); the union is not uniformly better. |
| **F3** | PASS | Hybrid 1 vs none 0 at 1/day; loss −1; at 2/4/day hybrid 1 vs none 1. | The criterion is "state conditioning normalizes the fault" (hybrid ≤ none − 2). It does not. With only one detection total, this test has almost no power; it is a pass by absence of a 2-event loss, not a demonstration of benefit. |
| **F4** | PASS | OR tail: 0.5/day +12.2%, **1/day −43.9%**, 2/day +16.2%. | Passes by 6.1 points at the primary budget. A and AND fail the same 50% limit at 1/day (+52.3%); weighted is +48.2%. The pass is method-specific and fragile. |
| **F5** | PASS | 25 OR alarms at 1/day; 0 have `n < 200`; 0 have `n < 50`. | Passes the 25% rule at the tested budget. Hidden risk: at 4/day, **95 of 263 B alarms (36.1%)** use `state_low` with `n < 200`. F5 only inspects 1/day, so it does not certify higher budgets. |
| **F6** | PASS | Maximum masked share 8.5052% (event 3). Event 1 3.40%, events 2/4 0%. | Passes the 10% limit by 1.49 points. Separate problem: event 1's detection-credit window is 35.2% masked, so event-1 recall is structurally under-measured. |
| **F7** | PASS | Median Spearman 0.385 (0.588, 0.179, 0.289, 0.482); top-20 Jaccard 0. | Both conditions fail: ρ < 0.8 and F1 holds. Scores are genuinely different. |
| **F8** | PASS | OR recall 2 vs B4 1 at every budget; mean abs budget error across budgets OR 1.599 vs B4 3.266. | Passes on both clauses. B4's event-4 detection is real but comes with 22 alarms at 1/day and much worse budget error. |
| **F9** | PASS | Primary vs hygienic recall at 0.5/1/2/4: 1/2/2/2 vs 1/2/2/3; differences 0, 0, 0, +1. | Passes the "≥2 events" rule. The +1 at 4/day is worth noting: hygiene changes a high-budget result, so the 4/day row is not hygiene-stable. |

No criterion was falsified. The weakest passes are F3 (near-vacuous power), F4
(6.1-point margin, other methods fail), F5 (only 1/day inspected, higher-budget
risk visible), and F2 at 4/day (union loses to B).

---

## 6. Real vs synthetic vs hypothesis vs limitation

### 6.1 Real measurements

- Every number in §4 and §5 comes from the real MetroPT-3 run, written to
  `reports/experiments/` by `scripts/run_experiments.py`.
- Data-layer facts (1,516,948 native rows, 306,960 grid rows, 252,720 observed,
  321 segments, four events) are recomputed by the data layer and asserted in
  `tests/test_real_data.py`.
- Mask shares, fold counts, observed days, alarm counts, calibration levels and
  `n` are produced by code, not typed by hand.

### 6.2 Synthetic tests (properties, not real-data evidence)

- Score A monotonicity under an injected residual, channel identification,
  arm distinctness, train-only fit, future-poison invariance.
- Score B survival monotonicity in run length, independence from Score A.
- Calibration direction, cell→state→global back-off, `state_low`,
  `n<50` refusal, chronological split, reference-only fit.
- Debounce spacing, gap handling, refusal, budget-solver accuracy and
  monotonicity, unreachable/coarse reporting.
- Fusion algebra, refused-side semantics, λ grid coverage.
- Reproducibility: two identical real fold-1 runs are byte-identical.

These tests prove the implementation matches the specification. They are **not**
evidence about MetroPT-3 detection performance.

### 6.3 Hypotheses (plausible, not established)

- Score A catches event 3 because the five-channel value envelope changes
  before/at the event; the context arm improves it (state/hybrid 1/4 vs none
  0/4) because the event occurs in a particular operating context.
- Score B catches event 4 because loaded runs lengthen around the event; it
  misses event 3 because event-3 runs are not unusually long relative to the
  reference.
- Global z catches event 2 because event 2 is a short, sharp value excursion; A
  and B are smoothed/state-based and miss it.
- The fold-4 run-length flood is caused by Score B's discrete p-value floor
  (`7.5e-5` for the loaded state), not by a genuine 5.37/day alarm rate.

Each hypothesis is consistent with the numbers but was not tested as a separate
causal experiment.

### 6.4 Limitations

1. **Four events, 7.746 observed event-fold days.** No significance claim, no
   confidence interval that would survive multiple comparisons. Recall "2/4" is
   a count, not an estimate of a population rate.
2. **Budget coarseness.** Score B's realized rate is effectively
   {0, ~1.4, ~4.5}/day on the tail; 1/day and 2/day produce the same 35 alarms.
   OR is similarly flat between 0.5 and 1/day (14 alarms both). Fine-grained
   budget control is not demonstrated.
3. **Fold-4 loaded calibration is thin.** A state 41 rows, B state 30 rows; both
   below 50, so loaded rows use global calibration. Event 4's detection is not
   evidence that loaded-state calibration works.
4. **Event 1 detection credit is 35.2% masked.** 169 of 480 credit minutes cannot
   be credited. Event-1 recall is under-measured relative to events 2–4.
5. **Tail control is not tight.** OR 0.56/day at a 1/day request (−43.9%); A and
   AND exceed +50%; weighted is +48.2%. Only OR passes F4, narrowly.
6. **Per-cell support collapses above 1/day.** One-row cells and 10-row
   `loaded|weekend` cells produce deviations up to 1.0 at 2–4/day. The aggregate
   budget can look acceptable while a cell is effectively uncalibrated.
7. **Score B refuses test rows.** 9 rows in fold 1 and 10 in fold 3 cannot alarm
   at all. The detector is silent there by design, not because the plant is
   normal.
8. **Masks affect false alarms.** M-none floods fold 1 (20 false alarms, event 1
   detected only by the flood). Results depend on the verified mask layer.
9. **No finite-sample distribution-free guarantee.** The series is not
   exchangeable. "Calibration" here means empirical false-alarm control on the
   observed data.
10. **C1–C10 are deviations from the original protocol.** They are documented
    before the full run and are not silent changes; see §7.

---

## 7. Protocol deviations C1–C10

All are recorded in `docs/experimental_protocol.md` §11 with reasons and costs.

| ID | Deviation | Effect on this report |
| --- | --- | --- |
| C1 | Chronological 2/3 reference / 1/3 threshold selection within calibration. | Reference and α selection are separated; selection `n` is smaller. |
| C2 | Score B B2 not implemented; B1 primary, B3 secondary. | No cycle-period result. |
| C3 | OR Bonferroni is reporting-only. | Same alarm set at fixed budget; no significance claim. |
| C4 | Weighted fusion primary λ=0.5; full grid reported, not tuned on test. | λ=0.5 underperforms λ=0.25/0.0 but was fixed in advance. |
| C5 | B3 leave-one-run-out is approximate. | B3 is an ablation, not primary. |
| C6 | Rolling robust z uses exact trailing 6 h median/MAD, expanding first 359 min. | Baseline B2 definition. |
| C7 | M-none bypasses masks in split materialisation only; M-hygienic removes long loaded runs. | Ablation only; primary masks unchanged. |
| C8 | Baselines use the same Mondrian hierarchy and budget solver. | Fair alarm-layer comparison. |
| C9 | Score A Ridge α=1.0, lags 1/5, means 15/60/360, MAD 60, EWMA half-life 1 h, floor at train median, max over channels. | Primary Score A definition. |
| C10 | Score B conformal reference rows are completed reference runs only. | Removes censored-run artifacts; fold-4 loaded reference is small. |

No test-set threshold, arm, cell, λ or budget was selected after seeing test
results. The fold-4 loaded back-off is a consequence of the pre-registered
`n_refuse=50` rule, not a post-hoc choice.

---

## 8. Reproducibility and tests

### 8.1 Commands

```bash
# full real experiment (~5 min)
.venv/bin/python scripts/run_experiments.py --output reports/experiments

# regenerate all figures from frozen artefacts
.venv/bin/python scripts/make_figures.py

# full test suite
.venv/bin/python -m pytest tests/ -q
```

### 8.2 Frozen identifiers

| Item | Value |
| --- | --- |
| Config hash | `06954d34ced38410` |
| Dataset fingerprint | `420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed` |
| Python | 3.14.10 |
| pandas / numpy / sklearn | 2.3.3 / 2.5.4 / 1.9.1 |
| Budgets | 0.5, 1.0, 2.0, 4.0 |
| Debounce | 30 min |

### 8.3 Test result

```
141 passed in 87.34s
```

Day 3 tests added 35 tests in five files:

- `tests/test_score_direction.py` (7): monotonicity, channel identification,
  arm distinctness, train-only fit, future poison, B monotonicity, A/B
  independence.
- `tests/test_calibration_audit.py` (9): direction, cell/state/global back-off,
  `state_low`, refusal, missing keys, chronological split, reference-only fit,
  masked rows.
- `tests/test_alarm_budget.py` (8): debounce, gaps, refusal, null accuracy,
  monotonicity, unreachable/coarse, eligible rows, calibration-only output,
  debounce ≠ hysteresis.
- `tests/test_fusion.py` (5): OR/AND/weighted algebra, Bonferroni, refused side,
  NaN handling, λ grid.
- `tests/test_experiment_reproducibility.py` (6): frozen config hash, two
  identical fold-1 runs, real mask/label exclusion, exposure, masked event
  minutes.

Figure determinism was checked by running `scripts/make_figures.py` twice and
comparing SHA-256 hashes of all ten PNGs: identical.

Full-run determinism was checked by running the complete experiment a second
time to a separate directory and comparing SHA-256 hashes. `fold_results.csv`,
`alarms.csv`, `summary.json`, `falsification.json` and all four prediction files
are byte-identical; config hash, dataset fingerprint, budgets, debounce and
library versions match.

### 8.4 Artefacts

```
reports/experiments/
  run_manifest.json
  alarms/alarms.csv
  metrics/
    primary_results.csv          fold_results.csv
    context_arms.csv             context_diagnostics.csv
    tail_results.csv             score_b_variants.csv
    cell_calibration.csv         calibration_cell_sizes.csv
    score_relationship.csv       weighted_lambda.csv
    mask_coverage.csv            mask_ablations.csv
    event_outcomes.csv           falsification.json
    summary.json
  predictions/fold_{1..4}.csv
  figures/
    event_traces.png             budget_recall.png
    tail_budget_control.png      calibration_reliability.png
    context_arms.png             score_scatter.png
    score_contribution.png       weighted_lambda.png
    mask_coverage.png            event_delays.png
```

---

## 9. Figure guide

| Figure | What it shows | Main reading |
| --- | --- | --- |
| `event_traces.png` | p-values A/B/OR in each event test window, with detection-credit and event spans; OR alarms at 1/day. | Fold 3 has a clear A dip; fold 4 has a B p-floor run. |
| `budget_recall.png` | Recall and mean absolute budget error across budgets. | OR leads recall; B's rate is stepwise. |
| `tail_budget_control.png` | Realized vs requested rate on the tail. | OR tracks best; A/AND overshoot at 1/day. |
| `calibration_reliability.png` | Per-cell nominal α vs empirical FPR; calibration support at 1/day. | Most mass is near the diagonal; small cells scatter. |
| `context_arms.png` | Recall and false alarms by context arm. | State/hybrid gain event 3; no flood. |
| `score_scatter.png` | −log10 pA vs −log10 pB, detection rows highlighted. | Weak relationship; detection rows are not on a single diagonal. |
| `score_contribution.png` | Unique detections and which score fired OR alarms. | Event 3 A-only, event 4 B-only; OR alarms 11 A / 14 B including tail. |
| `weighted_lambda.png` | Recall and budget error across λ. | λ=0.5 is not optimal; λ=0.25 finds 2/4 at 2/day. |
| `mask_coverage.png` | Masked share of each event window. | Event 3 at 8.51%, under the 10% line. |
| `event_delays.png` | First-alarm delay by method and event at 1/day. | A early on event 3; B/OR late on event 4; baselines early. |

---

## 10. Verdict and conditions

**CONDITIONAL GO** for the following configuration:

> **Score A `hybrid` + Score B `B1`, fused by OR, budget 1/day, 30-minute
> debounce, Mondrian `(state, day type)` calibration with state→global back-off,
> masks and train-only fitting as implemented.**

Why GO: the full pipeline runs end-to-end on real data; 141 tests pass; F1–F9
all pass; the union is the only primary method with 2/4 recall at 1/day; its false-alarm
volume is the lowest among the 2/4 methods; the two scores are demonstrably
non-redundant; and the result is reproducible from a frozen config.

Why CONDITIONAL, not unconditional GO:

1. **Do not claim tight 1/day control.** OR realizes 0.56/day on the tail and
   1.42/day pooled on the event folds; 0.5 and 1/day are the same 14 tail
   alarms. If a hard rate is required, use a **hard quota** or report the
   empirical range, and do not present the solver as a proportional controller.
2. **Fix or explicitly abstain on fold-4 loaded rows.** A's loaded state has 41
   reference rows and B's has 30, both below 50; all loaded rows use global
   calibration. Before deployment, either widen the loaded-state reference or
   mark loaded-state alarms as low-support.
3. **Report event-1 recall with the masked-credit caveat.** 169/480 detection
   credit minutes are masked. Event 1 is not a fair test of the detector.
4. **Use 1/day, not 2–4/day, for any calibrated claim.** At 4/day, 36.1% of B
   alarms use `n < 200`, per-cell deviations reach 1.0 in one-row cells, and OR
   loses to B alone on recall. F5 does not certify those budgets.
5. **Keep the mask layer.** M-none floods fold 1 with 20 false alarms. The
   result depends on the verified masks.
6. **State the sample size.** Four events and 7.746 observed event-fold days;
   no significance, no distribution-free guarantee, and F3 is low-power.

No Streamlit app was part of the Day 3 experiment. The released viewer is a
read-only presentation layer over these frozen artefacts; it does not refit,
retune or change any number in this report.
