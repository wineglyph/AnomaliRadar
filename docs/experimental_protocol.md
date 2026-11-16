# Experimental Protocol — AnomaliRadar (Day 1)

**Status:** pre-registered design. Thresholds here are fixed **before** any results
are seen. Changing one after seeing results requires an entry in the changelog at
the end of this file and an explicit note in the README.

**Companion documents:** `docs/methodology.md` (definitions), `docs/day0_report.md`
(data facts).

---

## 1. Non-negotiable constraints

1. **4 labelled events.** No hypothesis tests on recall differences. Report exact
   counts, and where an interval is useful, a Clopper–Pearson interval labelled as
   very wide.
2. **No point-level optimisation.** Point PR-AUC may be reported only as a
   secondary diagnostic with the caveat that 1.71% of time is positive and one
   fault contributes thousands of positive rows.
3. **No test-set tuning.** Every hyperparameter, threshold, mask, and cell boundary
   is fixed on train/calibration. The test segment is touched once per fold.
4. **No relabelling.** Unlabelled episodes stay unlabelled; they are handled by
   sensitivity analysis, not by editing labels.
5. **Same data, same budget, same code path** for every method compared.

---

## 2. Temporal splits (rolling origin)

Each event gets its own fold. Nothing after an event's start is used to fit or
calibrate that event's model.

| Fold | Event | Train (fit) | Calibrate | Embargo | Test window |
| --- | --- | --- | --- | --- | --- |
| 1 | #1 | 02-01 00:00 → 03-21 00:00 | 03-21 00:00 → 04-11 00:00 | 6.5 d | 04-17 12:00 → 04-20 00:00 |
| 2 | #2 | 02-01 00:00 → 05-01 23:30 | 05-01 23:30 → 05-22 23:30 | 6.5 d | 05-29 11:30 → 05-31 06:00 |
| 3 | #3 | 02-01 00:00 → 05-08 10:00 | 05-08 10:00 → 05-29 10:00 | 6.5 d | 06-04 22:00 → 06-08 14:30 |
| 4 | #4 | 02-01 00:00 → 06-17 14:30 | 06-17 14:30 → 07-08 14:30 | 6.5 d | 07-15 02:30 → 07-16 19:00 |
| **T** | false-alarm control | fold-4 model, frozen | — | — | **08-01 00:00 → 09-01 03:59** (31 d) |

Rules:

- Train and calibration are **disjoint**; calibration is the final 21 days before
  the embargo.
- Test windows extend 12 h before the label start (for context plots and for the
  score trace), but **detection credit** is only awarded inside
  \([t^{start} - 2\text{ h},\; t^{start} + \tau_k]\). The wider window exists to
  compute detection delay and to avoid edge artefacts; it does not inflate recall.
- Observed onset lead/lag relative to label start: **−12.3 min (event 2),
  −8.5 min (event 3), −1.8 min (event 4), +24.0 min (event 1)**. The 2 h early
  tolerance is ~10× the observed lead, which is deliberate: the label is a
  maintenance report and its timestamp is not an onset detector.
- Fold 4's model is **not** refitted for the 31-day tail. This is the only honest
  false-alarm measurement, and it is a genuine out-of-sample budget test.
- Calibration is a fixed 21-day window (not "everything before") so folds are
  comparable.

### 2.1 Calibration contamination (known, measured, not hidden)

| Fold | Contaminating episodes inside calibration |
| --- | --- |
| 1 | 796-min loaded run on 03-28 (unlabelled, `Reservoirs` −0.37 bar) |
| 2 | 99.5-min run on 05-19 (unlabelled, −0.98 bar, looks like a leak) |
| 3 | none ≥ 30 min identified |
| 4 | none ≥ 30 min identified |

Primary analysis keeps them. A **sensitivity arm** ("hygienic calibration") drops
loaded runs > 2 h from the calibration segment and re-runs; if the two disagree on
event recall by ≥ 2 events, the result is reported as unstable and no claim is made.

---

## 3. Leakage controls

Implemented as tests, not as discipline. Each item below becomes an assertion in
`tests/` during Day 2.

| # | Invariant | Test |
| --- | --- | --- |
| L1 | `max(train_ts) < min(calib_ts) ≤ max(calib_ts) < min(test_ts)` | range assertions per fold |
| L2 | Embargo ≥ 6 days of raw buffer between calibration end and test start **and** calibration ends exactly 7 days before the labelled event start | fold table check |
| L3 | No feature at \(t\) reads any row with timestamp > \(t\) | poison-future: corrupt all rows after \(t\), features at \(t\) unchanged |
| L4 | Mask is a pure function of the sensor frame, independent of labels and fitted statistics; applied unchanged to every split | mask-signature equality under two different label sets |
| L5 | Masked rows appear in no fit set and no calibration set | set intersection == ∅ |
| L6 | Labels never enter fitting, calibration, or budget solving | function signature has no `y` argument; grep test |
| L7 | Regime used as context is \(s_{t-1}\), never \(s_t\) | column-shift test |
| L8 | Alarm threshold \(\alpha\) depends only on calibration data | recompute with test lengthened → identical \(\alpha\) |
| L9 | Gap boundaries reset state; no statistic spans a > 1 h gap | constructed gap fixture |
| L10 | Scaler/normaliser statistics fitted on train only | fit-on-train fixture |

A fold that violates any invariant is **invalid**; the protocol does not "fix it up".

---

## 4. Baselines

All baselines produce a per-minute score, then pass through the **same** alarm layer
(debounce 30 min + budget solver) so that budget is held constant.

| ID | Method | Definition | Context |
| --- | --- | --- | --- |
| B1 | Global robust z | \(\max_c |y_{c,t} - \mathrm{med}_c| / (1.4826\,\mathrm{MAD}_c)\), stats from train | none |
| B2 | Rolling robust z | same, median/MAD over trailing 6 h, causal | none |
| B3 | Isolation Forest (raw) | 5 value channels at \(t\), `n_estimators=200`, fit on train | none |
| B4 | Isolation Forest + context | B3 plus hour, day type, \(s_{t-1}\), \(r_{t-1}\) | lagged state + calendar |
| P1 | Score A alone | §5 methodology, best context arm | per arm |
| P2 | Score B alone | §6 methodology | state |
| P3 | **Union (proposed)** | \(\min(p^A, p^B)\) | lagged state + day type |

Notes:

- B3/B4 use the same 5 channels as Score A so that the comparison isolates the
  method, not the input set.
- B4 is the critical "naive context-aware" baseline: if P3 cannot beat it, the
  two-score machinery is not paying for itself.
- No deep learning baselines. Out of scope by constraint, and 4 events cannot
  support them.

---

## 5. Ablation matrix

Every row is a full run of the same pipeline; only the listed factor changes.

### 5.1 Context arms (tests the original hypothesis)

| Arm | Context used by Score A | Expected outcome (hypothesis, not result) |
| --- | --- | --- |
| A0 | none (global scale) | reference |
| A1 | calendar only | ≈ A0, because η²(hour) = 0.0017 |
| A2 | lagged state only | better calibration, risk of normalization |
| A3 | lagged state + calendar | primary candidate |

A1 is a **null control**: if it beats A0 materially, the Day 0 variance
decomposition was misleading and must be re-examined.

### 5.2 Score ablations

| Arm | Scores | Purpose |
| --- | --- | --- |
| S-A | A only | value channel alone |
| S-B | B only | state channel alone |
| S-AB | A + B, OR | proposed |
| S-AB-and | A + B, AND | precision-oriented control |
| S-AB-w | A + B, weighted, λ grid | fusion comparison |

### 5.3 Mask ablations

| Arm | Change | Purpose |
| --- | --- | --- |
| M-none | no M1/M2 masks | quantifies the cost of ignoring data quality |
| M-primary | M1 + M2 + M3 (default) | reported configuration |
| M-hygienic | primary + drop > 2 h loaded runs from calibration | calibration robustness |

### 5.4 Score B variants

B1 (survival, primary), B2 (cycle-period deviation), B3 (hazard). All three at the
same budget, same test windows.

---

## 6. Metrics

Let \(E_k = [t^{start}_k, t^{end}_k]\) be event \(k\), \(\delta_{lead}=2\) h,
\(\tau_k = \min(6\text{ h}, \tfrac14|E_k|)\), and \(\mathcal{A}\) the emitted alarms.

### 6.1 Event recall (primary)

\[
\text{Recall} = \frac{1}{4}\sum_{k=1}^{4} \mathbb{1}\left[\exists a \in \mathcal{A}:\
a \in [t^{start}_k - \delta_{lead},\; t^{start}_k + \tau_k]\right]
\]

Reported as `x/4` with the four individual booleans, never as a percentage alone.

### 6.2 Detection delay

For a detected event: \(\Delta_k = a^*_k - t^{start}_k\), where \(a^*_k\) is the
**first** alarm in the window. Negative values (early alarm) are legitimate and
reported as such. Report the four values individually; a mean over 4 is not
meaningful.

### 6.3 Alarm volume

- **Total alarms/day** = \(|\mathcal{A}| / T_{obs}\), where \(T_{obs}\) excludes
  masked and gap time.
- **False alarms/day** = alarms outside all dilated event windows / \(T_{obs}^{outside}\).
- **Budget error** = \((\text{realized} - B) / B\), reported on both calibration and
  test. Sign matters: over-alarming is worse operationally than under-alarming here,
  because the budget is the product promise.

### 6.4 Calibration quality

| Metric | Definition |
| --- | --- |
| Per-cell empirical FPR | fraction of non-event observations in cell \(g\) whose fused p ≤ \(\alpha\), vs the nominal rate implied by \(\alpha\) |
| Max cell deviation | \(\max_g |\widehat{FPR}_g - \alpha|\) |
| Cell sizes | \(n_g\) per cell, reported as a table |
| Back-off usage | distribution of back-off depth over emitted alarms |
| Uncalibrated alarms | count of alarms where \(n_g < 50\) (must be 0) |

### 6.5 Secondary (diagnostic only)

Point PR-AUC and ROC-AUC per fold, explicitly labelled:

> "These are dominated by the largest event (event 3, 17,315 rows) and by alarm
> clustering; they are not evidence of operational quality."

### 6.6 Budget levels

Evaluate at \(B \in \{0.5, 1, 2, 4\}\) alarms/day. Report the full grid, not the
best cell. The headline number is \(B=1\) (one alarm per day), chosen before
running because it is the only level where 4 events in ~200 days is not swamped by
false alarms.

### 6.7 Budget comparability rule

A method is only compared at a budget it can actually achieve on calibration. If a
method's floor exceeds \(B\), it is reported as `infeasible at B` and excluded from
that column — never silently assigned a different budget.

---

## 7. Execution order (Day 2–3)

Fixed sequence; each step gates the next.

1. **Data layer** — loader, 1-min grid, M1–M3 masks, hysteresis state.
   Gate: L1, L3, L4, L9 pass; M1 share ≈ 3.3% of rows; gap share ≈ 17% of wall clock.
2. **Score A** — features, model, causal scale. Gate: L3, L7, L10 pass; residual
   diagnostics sane on fold 1 calibration.
3. **Score B** — run-length extraction, survival statistic. Gate: run-length
   distribution is in the expected order of magnitude (off ≈ 10–15 min,
   offloaded ≈ 7 min, loaded ≤ 5 min after hysteresis) and its tail is
   non-degenerate.
4. **Calibration + budget** — Mondrian cells, back-off, solver, debounce.
   Gate: on synthetic exchangeable data, realized rate ≈ requested rate within
   binomial noise (unit test).
5. **Baselines** — B1–B4 with identical alarm layer.
6. **Ablations** — §5, one factor at a time.
7. **Tail false-alarm run** — fold-4 model on the 31-day tail.
8. **Reporting** — tables auto-generated from `results/`, no hand-typed numbers.

If step 3 shows Score B has no usable dynamic range (e.g. loaded runs are already
almost all ≤ 3 min outside events and the tail is too thin), stop and report
`Score B infeasible` rather than forcing it.

---

## 8. Pre-registered falsification criteria

The project fails its own claim if **any** of the following holds. Thresholds are
fixed now.

| ID | Criterion | Threshold |
| --- | --- | --- |
| **F1** | Two-score union has no unique detections | 0 events detected by exactly one score that the other misses |
| **F2** | Union does not improve recall over the best single score | no gain at ≥ 2 of the 4 budget levels |
| **F3** | State conditioning normalizes the fault | A2/A3 recall ≤ A0 recall − 2 events |
| **F4** | Budget is not controllable | realized/day differs from \(B\) by > 50% on the 31-day tail at \(B \le 2\) |
| **F5** | Calibration cells unreliable | > 25% of emitted alarms have \(n_g < 200\) after back-off |
| **F6** | Masks destroy the labels | masked time inside any event window > 10% of that window |
| **F7** | Scores are redundant | Spearman \(\rho(a,b) > 0.8\) **and** F1 also holds |
| **F8** | Union loses to naive context-IF | P3 recall < B4 recall at every budget, with better budget error |
| **F9** | Results unstable under calibration hygiene | hygienic vs primary differ by ≥ 2 events |

Current status of the measurable ones from Day 0 (checked, not assumed):

- **F6: PASS, 1.49-point margin, re-checked on Day 2.** Event 3's combined mask is
  **8.51%** of its window (M1 0, M2 7.55%, M3 0.95%); event 1 is 3.40%, events 2
  and 4 are 0%. The earlier 7.2% figure counted only the fully-inside gap and
  omitted the 0.189 h tail of a second gap. Event 1's 19 min of M1 is a corrected
  timestamp arithmetic, not a new mask.
- **F5: at risk** — 24% of fine cells are < 200 rows; this is why the primary
  granularity is 6 cells, not 144. Day 2 confirms fold 4's `loaded` cells (48 + 29
  = 77 rows) require state-level back-off.
- F1–F4, F7–F9: untested; require implementation.

### 8.1 Fallbacks if falsified

| Failure | Fallback design | What the README then says |
| --- | --- | --- |
| F1 / F2 / F7 | Single-score: state-persistence only if it detects more, else value only | "Two-score composition did not add measurable value; we report the single-score variant." |
| F3 | Drop state conditioning from Score A; use A0/A1 | "State conditioning normalized the fault; context is reported as a negative result." |
| F4 | Replace p-value threshold with hard rate limiting (quota per day) | "Budget enforced by quota; statistical calibration claim withdrawn." |
| F5 | Coarsen cells to state-only (3 cells) | "Hour-level calibration was not supported by the data." |
| F6 | Switch primary dataset to NAB | "MetroPT-3 masks overlapped labels; dataset changed." |
| F9 | Widen calibration, or report range instead of point estimate | "Results sensitive to calibration contamination." |

Note the fallbacks are **simpler**, not more complex. That is deliberate: the
failure mode of a small project should be "we shipped a smaller honest thing", not
"we added components until the numbers moved".

---

## 9. Reporting format

Every result table must carry:

1. fold id, test window, budget \(B\), debounce \(\delta\);
2. the exact git commit and config hash;
3. observed exposure hours and masked share;
4. the four per-event booleans, not just the sum;
5. a `caveat` column referencing the applicable limitations.

No figure or number may be typed by hand into the README; tables are generated.

---

## 10. Reproducibility checklist

- [ ] fixed seeds for IForest, feature sampling, budget bisection tie-breaks
- [ ] pinned dependency versions
- [ ] config file per run in `configs/`, hash recorded
- [ ] masks written to `data/processed/masks.parquet` + hash
- [ ] fold boundaries derived from a single `folds.yaml`, not duplicated
- [ ] `pytest -q` green, including L1–L10
- [ ] one-command reproduction: `python scripts/run_experiments.py --config ...`

---

## 11. Changelog

| Date | Change | Reason |
| --- | --- | --- |
| 2026-11-16 | initial protocol | Day 1 |
| 2026-11-16 | **L2 reinterpreted.** The literal "≥7 days between calibration end and test start" conflicts with the pre-registered 12 h pre-event test window: `calib_end = event_start − 7 d`, so the raw buffer to `test_start = event_start − 12 h` is 6.5 d. L2 is now the conjunction of (a) calibration closing a full 7 days before the labelled event (**the stronger, meaningful margin**) and (b) a raw embargo of ≥6 days (**24× the longest causal window, 6 h**). Fold T keeps the 23.4-day buffer. No fold table value changed; only the invariant text. | Day 2 implementation surfaced the arithmetic; silently weakening the check to "≥6 d" would have hidden it. |
| 2026-11-16 | **L4 clarified.** Masks are label-free rule-based quality filters, so they are computed once on the whole sensor series rather than "on the training segment". The invariant is now "mask is a pure function of the sensor frame"; the test asserts hash equality under a relabelled frame. | Day 2: masks use no labels and no fitted statistics, so segment-local computation is neither possible nor meaningful. |
| 2026-11-16 | **C1 — calibration reference/threshold split.** Within the fixed 21-day calibration window, the chronologically earliest 2/3 of eligible rows build the conformal reference distributions; the latest 1/3 is used only to select the alarm threshold α. Test rows are scored against the reference distributions. | Statistical audit requested in the final brief. Using one set of observations both to build the empirical tail and to choose α makes the in-sample budget estimate optimistic, especially at the 0.5/day budget. The split is chronological, involves no labels, and was fixed before any test-fold result was inspected. Cost: the selection set is smaller (~1/3 of calibration), so α is chosen with fewer observations; this is reported with every budget. |
| 2026-11-16 | **C2 — Score B B2 not implemented.** The pre-registered cycle-period variant (B2) is reported as *not implemented*, not as a failed method. B1 remains primary and B3 (discrete-time hazard) is the implemented secondary variant. | B2 requires a stable cycle-period estimator and the data contain many missing cycles and long shutdowns. Adding a fragile period estimator would change the primary experiment and could not be validated on four events. This is an omission, not evidence against B2. |
| 2026-11-16 | **C3 — OR correction is reporting-only.** The primary OR ranking uses \(u=\min(p^A,p^B)\). A Bonferroni-corrected column \(u'=\min(1,2\min(p^A,p^B))\) is also stored and reported. Because the correction is a monotone rescaling absorbed by the budget solver, the alarm set at a fixed budget is unchanged unless \(2\min(p^A,p^B)>1\). No significance claim is made from p-values. | Implements the brief's "OR/min-p with documented correction" while keeping the pre-registered primary ranking. |
| 2026-11-16 | **C4 — weighted fusion primary λ fixed.** Weighted fusion uses \(u=\lambda p^A+(1-\lambda)p^B\) with primary \(\lambda=0.5\) fixed before test. The full λ grid {0, 0.25, 0.5, 0.75, 1} is evaluated on the calibration selection set and reported as a sensitivity table; the test set is never used to choose λ. Missing/refused p-values enter as 1 for AND/weighted and are ignored by OR when the other score is available. | The methodology pre-registers a λ grid fitted on calibration only. Fixing the primary value avoids post-hoc selection; the grid remains visible. |
| 2026-11-16 | **C5 — Score B B3 leave-one-run-out is approximate.** B1 removes the scored run exactly from the empirical survival distribution. B3 rebuilds a smoothed hazard from all reference runs, so reference-row scores retain a small in-sample optimism. B3 is an ablation; B1 is primary. | Exact leave-one-run-out hazard rebuilding would require refitting the hazard for every reference run and is not needed for the primary result. The approximation is stated wherever B3 is reported. |
| 2026-11-16 | **C6 — rolling robust-z dispersion.** B2 uses the exact trailing 6 h median and the exact trailing 6 h median absolute deviation around that median, computed within segments on values shifted by one minute. The first 359 minutes of a segment use an expanding window. | Faithful implementation of the pre-registered causal rolling median/MAD, with no global statistic and no future row. |
| 2026-11-16 | **C7 — mask ablation M-none bypasses masks in split materialisation only.** The masks are still computed and reported; the ablation feeds the same pipeline a split built with `mask=False` everywhere. M-hygienic additionally removes loaded runs longer than 2 h that intersect the calibration window. | Quantifies the cost of ignoring data quality without changing the verified mask definitions or the primary analysis. |
| 2026-11-16 | **C8 — baseline calibration uses the same Mondrian hierarchy.** Baselines B1–B4 are converted to p-values through the same cell → state → global calibrator and the same budget solver as Scores A/B. | Protocol §1.5 requires the same alarm layer and budget. Giving baselines a different calibration path would confound method with calibration. |
| 2026-11-16 | **C9 — Score A implementation details fixed.** Ridge α=1.0 per channel; lags 1 and 5; rolling means 15/60/360 min; own-window MAD 60 min; causal EWMA scale of |e| with 1 h half-life shifted one minute and floored at the train median |e|; per-channel nonconformity log1p(|e|/scale); Score A = max over the five channels. Calendar = hour sin/cos + weekend; state = lagged-state one-hots + lagged run length + lagged 1 h transition count. | These are the simplest reproducible choices consistent with methodology §5. The scale floor prevents a near-zero EWMA from producing unbounded scores; log1p keeps one extreme residual from dominating the maximum. No same-row state is used. |
| 2026-11-16 | **C10 — Score B conformal reference rows are the completed reference runs.** The survival distribution \(\mathcal{D}_s\) uses completed runs wholly inside the reference part, as pre-registered. The conformal reference scores for Score B are restricted to rows belonging to those same runs (leave-one-run-out), rather than every row whose timestamp falls in the reference window. Rows from runs censored by the reference boundary are otherwise scored against a distribution that deliberately omits their own run, which manufactures artificial outliers in the conformal tail. Those rows remain scored as selection/test observations but do not define the reference. | Makes the Score B reference construction internally consistent with §6.1 and the leave-one-run-out convention. It is a calibration-set definition, not a threshold change, and was fixed before the full test run. |
