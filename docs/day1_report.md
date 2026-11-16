# Day 1 Report — Method Architecture and Experimental Protocol

**Date:** 2026-11-16
**Scope:** architecture and protocol only. No model code, no CCCS implementation, no UI.
**Deliverables:** `docs/methodology.md`, `docs/experimental_protocol.md`, this report.
**Dataset unchanged.** `docs/datasets.md` required no correction — the Day 0
statements about licence, schema, and the two documentation errors were re-checked
against the file and still hold.

All quantitative statements below were **re-computed from
`data/raw/metropt3/MetroPT3(AirCompressor).csv` on 2026-11-16**, not copied from
Day 0. Where a Day 0 number is reused, it was reproduced.

---

## 1. New evidence collected on Day 1

Day 0 established that the fault is a sustained state change. Day 1 needed three
further facts before the architecture could be fixed. All three came out against
the original design.

### 1.1 Calendar context carries almost no information

Variance of each channel explained by context (η², computed on 96.7% of rows after
removing the 7 frozen blocks):

| Context | `Reservoirs` | `TP3` | `Oil_temperature` | `Motor_current` | `H1` |
| --- | --- | --- | --- | --- | --- |
| hour | 0.0007 | 0.0007 | 0.0325 | 0.0006 | 0.0004 |
| day type | 0.0002 | 0.0002 | 0.0105 | 0.0022 | 0.0043 |
| **calendar (hour + day type)** | **0.0017** | **0.0017** | **0.0439** | 0.0033 | 0.0057 |
| **operating regime** | **0.4131** | **0.4132** | **0.3261** | 0.9966 | 0.9764 |
| regime + hour | 0.4173 | 0.4174 | 0.3591 | 0.9967 | 0.9765 |

Calendar explains **0.2%** of reservoir pressure variance. Adding hour on top of
regime adds **0.4 percentage points**. The per-hour noise scale is also flat —
median |Δ`Reservoirs`| varies by only 1.14× across the 24 hours.

**Consequence:** the project's original premise ("a value may be normal at one hour
and unusual at another") is **not supported by this dataset**. Calendar-conditioned
value modelling here would be statistically indistinguishable from a global model.
What actually varies is the machine's operating state.

This is a negative result about the hypothesis, not a bug in the analysis. It is
reported rather than papered over: the README must say "state-aware", not
"calendar-context-aware".

### 1.2 The two scores are likely to carry different information — but it is not proven

Across the four events the value signal is inconsistent:

| Event | Duration | `Reservoirs` Δ vs prior week | Max loaded run inside window | Pressure-based detection plausible? |
| --- | --- | --- | --- | --- |
| 1 | 24.0 h | **−0.17 bar** | **1168 min** | weak |
| 2 | 6.5 h | −0.38 bar | 308 min | moderate |
| 3 | 52.5 h | −0.96 bar | 131 min | strong |
| 4 | 4.5 h | −0.81 bar | 153 min | strong |

Event 1 is the discriminating case: its pressure deviation is small, but it
contains the **longest loaded run in the entire dataset** (19.5 h, versus a median
loaded run of 2 min). If Score B earns its place anywhere, it is here.

This is a **hypothesis**, and it is exactly what falsification criterion F1 tests.

### 1.3 Score B alone is not sufficient either

Loaded episodes ≥ 30 min: **77**, of which only **28** intersect a label. **13**
exceed 2 h. Examples:

| Start | Duration | `Reservoirs` Δ vs prior 3 d | Interpretation |
| --- | --- | --- | --- |
| 2020-03-28 23:23 | 796.5 min | −0.37 | ambiguous |
| 2020-04-12 11:50 | 711.8 min | **+0.44** | pressure *rises* → benign sustained demand |
| 2020-03-12 00:17 | 698.3 min | +0.40 | benign |
| 2020-05-19 23:28 | 99.5 min | **−0.98** | looks like an unlabelled leak |
| 2020-07-15 14:25 | 153.3 min | −0.05 | labelled event 4 |

So state persistence fires on ≈49 unlabelled episodes over 7 months (~0.23/day).
Some are benign, some may be real unlabelled faults. **Neither score alone is
obviously sufficient**, which is the honest argument for the combination — and also
the reason F1/F2 exist.

### 1.4 Structural details that changed the design

- **Event 3 fragments badly**: 143 *raw* loaded episodes intersect its 52.5 h
  window, because single-row transition blips split long runs. Raw episodes are
  unusable; hysteresis with a 3-minute minimum dwell is now mandatory, not an
  optimisation.
- **Onset lead time, measured at the label boundary**: the loaded state begins
  **12.3 min** (event 2), **8.5 min** (event 3) and **1.8 min** (event 4) before the
  label start, and **24.0 min after** it for event 1 (the compressor is `off` at
  00:00 and transitions at 00:23:59). The label timestamp therefore tracks the
  state change within half an hour, in both directions. The 2 h early tolerance and
  12 h context window are sized from this, not guessed.
- **Mask overlap re-verified**: 7 all-analog frozen blocks ≥ 1 h (3.32% of rows),
  **0 overlap** with any label. Event 3 contains **1** gap > 1 h totalling 3.78 h
  (**7.2%** of its window), leaving a 2.8-point margin under the 10% limit (F6).
  Note that 17.0% of the 213-day wall clock is absent due to nightly outages —
  a property of the machine's schedule, not a defect, and the reason false-alarm
  rates are normalised by *observed* days.

---

## 2. Answers to the eight questions

### Q1. Is the two-score architecture actually necessary?

**Provisionally yes, by design argument; not yet by evidence.**

- Necessity argument: event 1 has almost no pressure signature (−0.17 bar) but the
  longest loaded run in the dataset. A value-only score conditioned on state will
  see "loaded" as normal and miss it. A state-only score fires on 49 unlabelled
  episodes. Neither alone covers the label set.
- Evidential status: **unproven**. It becomes "necessary" only if F1 holds (at
  least one event detected by exactly one score) **and** F2 holds (union gains
  recall at ≥ 2 budget levels).

If F1/F2 fail, the architecture collapses to a single score and the report says so.
The design is a hypothesis with a pre-registered test, not a conclusion.

### Q2. Which context strategy should be primary?

**Lagged operating state × day type (6 cells). Calendar is a null control, not a
primary strategy.**

| Strategy | η² on values | Circularity risk | Cell density | Verdict |
| --- | --- | --- | --- | --- |
| Calendar (hour × day type) | 0.0017 | none | 48 cells, sparse at night | **null control arm** |
| State (same-row) | 0.41 | **severe** | dense | **forbidden** |
| State (lagged, hysteresis) | 0.41 at \(t-1\) | low | 6 cells, dense | **primary** |
| Hybrid (lagged state + calendar) | 0.41 | low | 48 cells after adding hour | secondary arm only |

Reason for preferring 6 cells over 144: calibration must have ≥ 200 samples per
cell, and 24% of hour-level cells fail that on this data. Adding hour buys 0.4
percentage points of explained variance and costs calibration reliability.

### Q3. How is regime prevented from normalizing the fault?

Five mechanisms, in order of strength:

1. **Separation of concerns (structural).** State is not a conditioning variable for
   its own channels. `Motor_current`, `DV_eletric`, `COMP`, `MPG` are excluded from
   Score A targets entirely (rule R1). State persistence gets its own score (B),
   where a change is the *signal*, not a nuisance to be explained.
2. **Lag (temporal).** Context uses \(s_{t-1}\) with hysteresis, never \(s_t\) (R2).
   A fault that begins at \(t\) cannot explain away the observation at \(t\).
3. **Measurement, not assertion.** Score A is run in four arms — none / calendar /
   lagged state / hybrid — and the **recall delta** between the no-context and
   state-context arms is reported (R3).
4. **Pre-registered failure condition.** If state conditioning costs ≥ 2 events of
   recall versus no context, criterion **F3** fires, state is removed from Score A,
   and the result is published as a negative finding.
5. **Exclusion of the near-deterministic channel.** `TP2` is 99.99% zero when off
   and 0.13% zero when loaded; it is a state indicator and is excluded (R4).

### Q4. How are calibration and the alarm budget implemented?

- **Per-channel scores** → `max` over 5 value channels for A; survival statistic for B.
- **Mondrian cells**: `(lagged state, day type)`, 6 cells, all dense.
- **Back-off**: `cell → state → global` when \(n_g < 200\); alarms below
  \(n_g=50\) are refused rather than emitted with a fake p-value.
- **Budget solver**: bisection over \(\alpha\) so that the *debounced* alarm rate on
  calibration equals \(B\) alarms/observed-day; `budget_unreachable` is reported
  explicitly when the floor exceeds \(B\).
- **Debounce** 30 min, so one incident produces one alarm.
- **Verification**: realized rate on the 31-day held-out tail (Aug 1 – Sep 1) with
  the fold-4 model frozen. F4 fires if realized differs from \(B\) by > 50%.
- **Honesty about validity**: exchangeability is false for this series; the claim is
  *empirically verified conditional false-alarm control*, not a finite-sample
  guarantee.

Because the budget fixes the alarm count, fusion is a **ranking** problem: a
Bonferroni factor on the min-p statistic is absorbed by \(\alpha\) and is therefore
irrelevant to operation. This is stated to avoid pretending the p-values have
significance-testing meaning.

### Q5. Which masking rules are mandatory?

| Mask | Rule | Effect |
| --- | --- | --- |
| M1 | all 7 analog channels constant ≥ 1 h → drop | 7 blocks, 3.32% of rows, max 51.4 h |
| M2 | timestamp gap > 1 h → break series, no interpolation | 160 gaps, 868.5 h |
| M3 | discard 30 min after any M1/M2 boundary | warm-up |
| M4 | `other` state excluded from calibration | 1.32% |

Plus: no forward-fill beyond 2 min; masks fitted on train only and applied unchanged
elsewhere; masked rows excluded from both false-alarm numerator and denominator.
Verified overlap: 0% for events 1/2/4; event 3 contains 1 gap > 1 h totalling 7.2% of its window — inside the 10% limit (F6), with a 2.8-point margin.

### Q6. In what order will experiments run?

Data layer → Score A → Score B → calibration/budget → baselines → ablations →
31-day false-alarm run → auto-generated reporting, with leakage tests gating each
step (`experimental_protocol.md` §7). If Score B shows no dynamic range at step 3,
stop and report it infeasible rather than tuning it into existence.

### Q7. What counts as failure?

Nine pre-registered criteria (F1–F9). The decisive ones:

- **F1**: zero events uniquely detected by either score → the second score is
  decorative.
- **F2**: no recall gain at ≥ 2 of 4 budgets.
- **F3**: state conditioning costs ≥ 2 events → normalization confirmed.
- **F4**: budget error > 50% on the tail at \(B \le 2\).
- **F6**: masks cover > 10% of an event window.

Each has a **simpler** fallback (single score, no state context, quota-based
limiting, coarser cells, or a dataset switch). Fallbacks never add complexity.

### Q8. What must be true before Day 2 coding?

1. This architecture is approved (especially Q1's provisional status and Q3's
   measurement-over-assertion stance).
2. The 9 fixed decisions in `methodology.md` §11 are frozen; changing one later
   needs a changelog entry and a README note.
3. Day 2 starts with the **data layer and leakage tests**, not with models. No
   modelling until L1, L3, L4, L9 pass.
4. Expectation is set: the primary deliverable may end up being the **calibration
   and budget audit**, not a recall win. That was already true at Day 0; Day 1 made
   it more likely.

---

## 3. What changed relative to the Day 0 plan

| Day 0 plan | Day 1 decision | Trigger |
| --- | --- | --- |
| One CCCS score, context-conditioned | Two scores, state separated out | circularity + state-change fault |
| Calendar context as the core idea | Calendar demoted to null control | η² = 0.0017 |
| Context = regime from same row | Lagged, hysteresis-confirmed state | same-row absorbs the fault |
| Threshold from a global α | Budget-solved α with debounce | budget is the product claim |
| Point-level metrics | Event-level only | 1.71% positive, 4 events |
| Fine context grid (regime × daytype × hour) | 6 cells primary, 144 as an arm | 24% of fine cells < 200 rows |

Nothing in the Day 0 **dataset** findings was changed. One Day 0 **design**
assumption — that calendar context would matter — was refuted by the data.

---

## 4. Honest assessment of the project's position

**Strengths.** The dataset is real, licensed, and rich; the fault signature is
unambiguous; there are 77 days of pre-failure normal data and a 31-day clean tail;
the calibration/budget story is measurable and genuinely useful to a practitioner;
the whole design is sklearn + pandas.

**Weaknesses, stated plainly.**

1. **Four events.** Any recall difference of one event is 25 percentage points. The
   project cannot demonstrate statistical superiority and should not try. What it
   *can* demonstrate is budget control and attribution — both measurable over
   thousands of hours, not four points.
2. **The original thesis is partly dead.** Calendar context does nothing here. The
   surviving thesis ("state-aware, calibrated, budgeted") is narrower and less
   exciting, but it is true and testable.
3. **Circularity is contained, not eliminated.** Lagging state reduces it; it does
   not remove the fact that state is derived from potentially faulty sensors. The
   ablation measures the residual effect but cannot prove its absence.
4. **Score B has a prior of failure.** 49 unlabelled persistent episodes is a lot;
   Score B may simply be a noisy detector of "compressor worked hard". The
   pressure-outcome analysis suggests some are real faults, but this is not
   established.
5. **Unlabelled positives** make both recall and precision estimates biased in
   opposite directions. This is unfixable with the available labels and must be in
   the README.
6. **One machine, one year, pandemic period.** No generalisation claim is possible.

**Net:** the project is worth doing, but its value is in the *audit* — a calibrated,
budgeted detector with honest error accounting — not in beating a baseline on AUC.
If that is not the deliverable the project wants, the honest alternative is to
switch to NAB, where calendar context is non-circular and there are 58 labelled
series. That would trade a real machine for statistical breadth.

---

## 5. Verdict

### CONDITIONAL GO

The architecture is coherent, implementable in the remaining time, and every claim
has a pre-registered test. It is conditional on four things:

1. **Approval of the narrowed thesis.** "State-aware, calibrate-then-budget" — with
   the calendar result published as a negative finding. If the project requires the
   original calendar-context claim, this dataset cannot support it and we should
   change datasets, not wording.
2. **Acceptance that the two-score architecture is provisional** until F1/F2 are
   measured. It may collapse to one score, and that outcome is a valid deliverable.
3. **Acceptance of event-level evaluation with n = 4.** No significance testing, no
   AUC claims, wide intervals.
4. **Day 2 starts with the data layer and leakage tests**, and modelling is blocked
   until those pass.

### Conditions that would flip this to NO-GO

- If approval requires a calendar-context benefit claim → **NO-GO on this dataset**;
  switch to NAB.
- If the 10% mask-overlap limit (F6) is exceeded once M1 is computed on the resampled
  grid — currently 7.2% for event 3, i.e. only a 2.8-point margin → **NO-GO**.
- If Score B's run-length distribution turns out to have no usable tail outside
  events (fails at protocol step 3) → the two-score design is dead; the project
  reduces to single-score value detection, which is **not** CONDITIONAL GO but a
  materially smaller project requiring re-approval.

---

## 6. Day 1 gate checklist

| Requirement | Status |
| --- | --- |
| Day 0 findings re-verified against the real file | done (η², masks, episodes, onset lead) |
| Two-score architecture specified with math | done — `methodology.md` §5–6 |
| Circularity analysed and mitigated | done — R1–R5, F3 |
| Three context strategies compared | done — §Q2, with measured η² |
| Masking and leakage rules specified | done — M1–M4, L1–L10 |
| Metrics defined | done — `experimental_protocol.md` §6 |
| Baselines enumerated | done — B1–B4, P1–P3 |
| Ablation matrix defined | done — §5.1–5.4 |
| Falsification criteria fixed in advance | done — F1–F9 |
| Execution order defined | done — §7, 8 gated steps |
| No model code written | confirmed |
| No dataset modification | confirmed |
| No fabricated results | confirmed — every number recomputed |
| Literature not presented as novel | confirmed — `methodology.md` §1.1 |

**Stopping here. Awaiting approval before Day 2.**
