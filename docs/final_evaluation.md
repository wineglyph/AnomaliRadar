# Final evaluation — AnomaliRadar on MetroPT-3

**Date:** 2026-11-16
**Scope:** Day 0–3, real MetroPT-3 data, frozen pre-registered protocol plus
deviations C1–C10.
**Config hash:** `06954d34ced38410`
**Dataset fingerprint:** `420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed`

---

## 1. Decision

# **CONDITIONAL GO**

The implementation is complete, reproducible and tested. On real data it detects
**2 of 4** labelled events at the primary 1/day budget with **1 false alarm**
across the four event folds, and all nine pre-registered falsification criteria
F1–F9 pass. It is **not** an unconditional GO because the budget response is
coarse, the fold-4 loaded-state calibration is below the refusal threshold, event
1's detection credit is one-third masked, and four events cannot support a tight
false-alarm guarantee.

**Recommended configuration:**

> **Score A `hybrid` + Score B `B1`, OR fusion by `min(pA,pB)`, 1 alarm per
> observed day, 30-minute debounce, Mondrian `(lagged state, day type)`
> calibration with cell → state → global back-off (`n_min=200`, `n_refuse=50`),
> chronological 2/3 reference / 1/3 threshold selection, masks M1–M3,
> train-only fitting, fold-4 model frozen for the tail.**

This is the only pre-registered primary configuration that reaches 2/4 at 1/day
while keeping false alarms at 1 and mean absolute budget error below 2.0.

---

## 2. What was evaluated

| Item | Value |
| --- | --- |
| Dataset | MetroPT-3, 1,516,948 native rows, 306,960 grid minutes |
| Observed grid | 252,720 minutes; 321 segments |
| Event folds | 4 chronological folds, one event each |
| Event-fold exposure | 7.746 observed days in total |
| Tail | fold-4 model frozen, 31 wall days, 24.958 observed days |
| Budgets | 0.5, 1.0, 2.0, 4.0 alarms/observed day |
| Primary budget | 1.0/day |
| Primary context arm | `hybrid` |
| Primary Score B | `B1` |
| Primary fusion | OR |
| Debounce | 30 min (state hysteresis is 3 min; separate) |
| Calibration | empirical Mondrian, not distribution-free |
| Test-set tuning | none |

No Streamlit app was part of the frozen experiment. The release adds a read-only
viewer over the exported artefacts; it does not refit or re-score. No threshold,
cell, arm, λ or budget was chosen from test results.

---

## 3. Headline real-data results

### 3.1 At 1/day (four event folds)

| Method | Recall | Events | Alarms | False | Alarms/day (pooled) | False/day (pooled) | Mean budget error |
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

- Event 3: A and OR alarm **10 minutes before** the labelled start.
- Event 4: B alarms at **+32 min**; OR's first accepted alarm is **+39 min**
  because the union threshold is stricter than B's standalone threshold, not
  because of the debounce.
- Events 1 and 2: missed by A, B, OR, weighted and AND. Global z catches event 2;
  no primary method does.
- The OR alarms are 11 fired by A and 14 by B across event folds and tail; the
  two scores are complementary, not redundant (median Spearman 0.385, top-20
  Jaccard 0).

### 3.2 Tail (fold-4 model, 24.958 observed days, no labels)

| Method | Requested | Realized/day | Relative error | Alarms |
| --- | ---: | ---: | ---: | ---: |
| **OR** | 1.0 | **0.5609** | **−43.9%** | 14 |
| A hybrid | 1.0 | 1.5225 | +52.3% | 38 |
| B1 | 1.0 | 1.4023 | +40.2% | 35 |
| Weighted | 1.0 | 1.4825 | +48.2% | 37 |
| AND | 1.0 | 1.5225 | +52.3% | 38 |
| Global z | 1.0 | 0.6811 | −31.9% | 17 |

Every tail alarm is false by construction (no labelled event in the window).
OR is inside the 50% limit at 1/day (−43.9%), as are B (+40.2%) and weighted
(+48.2%); A and AND exceed it (+52.3%). OR's margin is 6.1 percentage points.

---

## 4. F1–F9 verdicts

`PASS` means the pre-registered falsification criterion did not hold. A pass is
not proof; it is the absence of the specified failure.

| ID | Criterion | Verdict | Evidence | Strength |
| --- | --- | --- | --- | --- |
| **F1** | Union has no unique detections | **PASS** | Unique events at 1/day: 3 (A), 4 (B). At 2/day: 3, 4. At 4/day: 1, 4. | Strong on this data; depends on exactly two events. |
| **F2** | Union does not improve over the best single score | **PASS** | Gain +1 at 0.5, +1 at 1, +1 at 2, −1 at 4/day. Gain at 3 of 4 budgets. | Passes; loses at 4/day where B alone gets 3/4. |
| **F3** | State conditioning normalizes the fault | **PASS** | Hybrid 1/4 vs none 0/4 at 1/day; loss −1. Never loses 2 events. | Near-vacuous power: only one detection exists to lose. |
| **F4** | Budget not controllable on tail (≤2/day, >50%) | **PASS** | OR 0.5/day +12.2%, 1/day −43.9%, 2/day +16.2%. A/AND +52.3% at 1/day; weighted +48.2%. | Narrow: 6.1-point margin, method-specific. |
| **F5** | >25% of alarms have `n < 200` | **PASS** | 25 OR alarms at 1/day; 0 with `n < 200`; 0 with `n < 50`. | Narrow: only 1/day tested. At 4/day, 36.1% of B alarms use `n < 200`. |
| **F6** | Masks destroy labels (>10% any event) | **PASS** | Maximum 8.5052% (event 3). Event 1 3.40%, events 2/4 0%. | Passes by 1.49 points. Event 1 detection credit is 35.2% masked. |
| **F7** | Scores redundant (ρ > 0.8 and F1 holds) | **PASS** | Median ρ 0.385 (0.588, 0.179, 0.289, 0.482); top-20 Jaccard 0. | Strong. |
| **F8** | Union loses to context-IForest at every budget with better error | **PASS** | OR recall 2 vs B4 1 at every budget; mean abs budget error across budgets 1.599 vs 3.266. | Strong on the specified comparison. |
| **F9** | Hygiene changes results by ≥2 events | **PASS** | Primary vs hygienic at 0.5/1/2/4: 1/2/2/2 vs 1/2/2/3; differences 0,0,0,+1. | Passes; +1 at 4/day means the high-budget row is not hygiene-stable. |

**Overall: 9 PASS, 0 FAIL, `n_falsified=0`.**

### 4.1 What the passes do not mean

- **F1/F2 rest on two events.** Event 3 is A-only and event 4 is B-only. Remove
  either and the union has no unique detection. The result is real but fragile.
- **F3 has almost no power.** At 1/day only one event is detected by any arm, so
  the "lose two events" criterion cannot fire. It should not be cited as
  evidence that state conditioning is beneficial.
- **F4 passes only for OR.** A and AND exceed the 50% tail limit at 1/day. The
  budget claim is not a property of the pipeline; it is a property of the primary
  OR on this tail.
- **F5 tests one budget.** The pre-registered check is at 1/day. It says nothing
  about 2/day or 4/day, where `state_low` and small cells appear. Do not extend
  the claim.
- **F6 measures the event window, not the detection-credit window.** Event 1's
  event window is 3.40% masked, but its 480-minute detection-credit window is
  35.2% masked. Event-1 recall is under-measured.
- **No statistical significance.** Four events, no confidence interval that
  would survive multiplicity, no distribution-free guarantee.

---

## 5. Why this configuration, and not the alternatives

| Alternative | Result | Reason not selected |
| --- | --- | --- |
| Score A only | 1/4 at 1/day | Misses event 4; 2 alarms only. |
| Score B only | 1/4 at 1–2/day; 3/4 at 4/day with 150 alarms | 4/day is 16.7 alarms/day, unacceptable; 1/day misses event 3. |
| AND | 0/4 at every budget | A and B fire on different events; requiring agreement removes the useful detections. |
| Weighted λ=0.5 | 0/4 at 1/day; 1/4 at 4/day | Underperforms OR. λ=0.25 reaches 2/4 at 2/day but was not pre-registered as primary and is not selected post hoc. |
| Global robust z | 2/4 at 1/day | Same recall as OR but 7 false alarms vs 1 (0.904 vs 0.129 false/day) and it is a single naive statistic. |
| IForest + context | 1/4 at 1/day | 22 alarms (2.84/day), mean budget error 3.78; event-4 alarm is 101 min early but comes with a flood. |
| OR + Bonferroni | Same alarm set at fixed budget | The correction is a monotone rescaling absorbed by the solver; reporting-only, no significance claim. |

**Primary configuration:** `P1_A_hybrid`, `P2_B_B1`, `P3_or`, budget 1.0/day,
30-min debounce. This is the mechanical fallback implied by the pre-registered
comparison, not a post-hoc optimum.

---

## 6. Conditions attached to the GO

The system may proceed to the next phase only under these conditions.

### C-1. Do not claim tight 1/day calibration

OR realizes 0.56/day on the tail and 1.42/day pooled on event folds. 0.5/day and
1/day produce the **same 14 tail alarms**; Score B produces the same 35 tail
alarms at 1/day and 2/day. This is a step function, not proportional control.
If a hard rate is required, implement a **hard quota** (e.g. at most one alarm
per 24 h per stream) or report the empirical range. Do not present the
conformal p-value threshold as a guaranteed 1/day controller.

### C-2. Resolve or label fold-4 loaded-state calibration

Fold 4's loaded reference is below the refusal threshold:

| Score | Loaded cell rows | Loaded state rows | Behaviour |
| --- | ---: | ---: | --- |
| A hybrid | 19 + 22 | 41 | back-off to global (`n=13,382`) |
| B1 | 16 + 14 | 30 | back-off to global (`n=13,382`) |

Every fold-4 loaded alarm, including event 4's detection, uses **global**
calibration, not loaded-state calibration. Before deployment, either widen the
loaded reference (more data / longer calibration) or mark loaded-state alarms as
low-support and require operator review.

### C-3. Report event 1 with the masked-credit caveat

169 of 480 detection-credit minutes (35.2%) for event 1 are masked. Only 311
minutes can earn credit. Event 1 is not a fair test of the detector; a miss
there should not be counted as strong evidence against it.

### C-4. Treat 1/day as the ceiling for calibrated claims

At 4/day:

- 95 of 263 B alarms (36.1%) use `state_low` with `n < 200`;
- per-cell deviation reaches 1.0 in a one-row fold-3 cell and 0.30 in fold-1
  `loaded|weekend` (3 alarms in 10 rows);
- OR recall (2/4) is below B alone (3/4);
- the union is no longer the best method.

F5 does not certify these budgets. Use 1/day, or coarsen cells and re-validate
before going higher.

### C-5. Keep the verified mask layer

M-none floods fold 1 with 20 false alarms and "detects" event 1 only through
that flood. M-hygienic is essentially identical to primary. The result depends
on the masks; do not disable them for convenience.

### C-6. State the inferential limits in every report

Four labelled events, 7.746 observed event-fold days, one dataset, no
exchangeability. Use "empirical calibration" and "empirical false-alarm
control", never "distribution-free guarantee". F3 is low-power and F4 is
method-specific.

---

## 7. What would change the decision to NO-GO

The current verdict would become **NO-GO** if any of the following is a hard
requirement:

1. **Detect events 1 and 2.** A, B and OR miss both. Only global z and IForest
   raw catch event 2; no primary score does.
2. **Guarantee ≤1.0 alarm/day within 10%.** OR is −43.9% on the tail at 1/day.
   It is inside the pre-registered 50% limit but nowhere near 10%.
3. **Certify loaded-state calibration.** Fold 4 has 41 (A) / 30 (B) loaded rows,
   below the refusal threshold. The event-4 detection is global, not
   loaded-specific.
4. **Certify per-cell FPR at >1/day.** Small cells and `state_low` alarms break
   the cell-level claim even when the aggregate rate looks acceptable.
5. **Prove statistical significance / a distribution-free guarantee.** The
   design cannot; the sample is four events and the series is not exchangeable.

None of these is a failure of the implementation. They are limits of the
evidence.

---

## 8. Deliverables and reproduction

### 8.1 Documents

| File | Content |
| --- | --- |
| `docs/day0_report.md` | problem definition, dataset audit |
| `docs/day1_report.md` | feature/event investigation |
| `docs/day2_report.md` | data layer, masks, state, folds, leakage |
| `docs/day3_report.md` | scores, calibration, alarms, baselines, real metrics |
| `docs/final_evaluation.md` | this decision document |
| `docs/experimental_protocol.md` | binding protocol, F1–F9, C1–C10 |
| `docs/methodology.md` | Score A/B, calibration, alarm definitions |
| `docs/datasets.md` | dataset provenance |

### 8.2 Code

```
src/anomaliradar/model/
  config.py  common.py  score_a.py  score_b.py
  calibration.py  alarm.py  fusion.py  baselines.py
  evaluation.py  runner.py  __init__.py
scripts/
  run_experiments.py  make_figures.py
tests/
  test_score_direction.py  test_calibration_audit.py
  test_alarm_budget.py  test_fusion.py
  test_experiment_reproducibility.py
```

### 8.3 Artefacts

```
reports/experiments/
  run_manifest.json
  alarms/alarms.csv
  metrics/*.csv, metrics/*.json
  predictions/fold_{1..4}.csv
  figures/*.png (10 deterministic figures)
```

### 8.4 Commands

```bash
.venv/bin/python scripts/run_experiments.py --output reports/experiments
.venv/bin/python scripts/make_figures.py
.venv/bin/python -m pytest tests/ -q
```

### 8.5 Verification

| Check | Result |
| --- | --- |
| Test suite | **141 passed in 87.34 s** |
| Day 3 tests added | 35 across 5 files |
| Config hash | `06954d34ced38410` |
| Dataset fingerprint | `420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed` |
| Second full run | byte-identical fold results, alarms, summary, falsification and predictions |
| Figure determinism | identical SHA-256 on repeated runs |
| Falsification | 9 PASS, 0 FAIL, `n_falsified=0` |

---

## 9. Final statement

AnomaliRadar now has a complete, deterministic, leakage-audited real-data
pipeline: value-anomaly Score A with causal features and four context arms,
state-persistence Score B with segment-boundary resets and survival
nonconformity, empirical Mondrian calibration with explicit back-off and
refusal, a calibration-only budget solver, a 30-minute alarm debounce, five
fusion rules, four baselines, and a full F1–F9 audit.

On MetroPT-3 it finds **2 of 4** labelled events with **1 false alarm** at
1/day, using two genuinely different scores. It misses events 1 and 2, its
budget response is coarse, its fold-4 loaded calibration is thin, and its
false-alarm evidence rests on four events and 7.75 observed event-fold days.

That is enough for a **CONDITIONAL GO** with the conditions in §6. It is not
enough for an unconditional deployment claim, and the report should not pretend
otherwise.
