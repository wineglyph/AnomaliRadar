# Methodology — AnomaliRadar (Day 1)

**Status:** design specification. No implementation yet.
**Supersedes:** the single-score CCCS sketch in the Day 0 proposal (see §2 for why).
**Related:** `docs/day0_report.md`, `docs/datasets.md`, `docs/experimental_protocol.md`

---

## 1. Scope and claim

AnomaliRadar produces, for each observation of a multivariate sensor stream:

1. a **calibrated anomaly score** with a stated false-alarm interpretation;
2. an **explanation** attributing the alarm to either a value deviation or a state
   persistence anomaly;
3. an operating point expressed as an **alarm budget** (alarms per observed day),
   not as a hand-tuned threshold.

**We do not claim accuracy superiority.** The claim under test is narrower:

> Given a fixed alarm budget, does splitting the anomaly signal into a value
> channel and a state-persistence channel — and calibrating each conditionally —
> detect more labelled events than a single score at the same budget?

This is falsifiable with 4 events (see `experimental_protocol.md` §8). If it fails,
the project reports the failure and falls back to a single-score design.

### 1.1 What is *not* novel

Conformal anomaly detection, Mondrian/group-conditional calibration, run-length
survival analysis, and residual-based detection are all established. `nonconform`
(arXiv 2605.13642) already packages conformal AD for sklearn/PyOD. Our contribution
is an **application-level composition** plus an audit trail, not a new algorithm.
This must appear in the README verbatim in spirit.

---

## 2. Why the original single-score design is rejected

Day 0 established three facts that break the original CCCS sketch:

| Finding | Consequence |
| --- | --- |
| Calendar explains **0.2%** of `Reservoirs` variance (η²=0.0017); operating regime explains **41.3%** | A calendar-conditioned value model is indistinguishable from a global one on this dataset |
| The fault signature is **loaded share 15% → 97–99%** for all four events | Conditioning a value score on same-row regime would explain the fault away |
| Regime is derived from `Motor_current`/`DV_eletric`, the very channels that change during faults | Regime is simultaneously context and symptom |

Therefore: **regime cannot be a neutral conditioning variable.** It has to be part
of what is being tested, i.e. its own score, and any use of it as context must be
lagged and must be validated for normalization explicitly.

The original "context-aware" framing survives only in a weaker, honest form:

> **State-aware** anomaly detection, with calendar context as a tested-and-rejected
> hypothesis rather than an assumed benefit.

Reporting that negative result is part of the deliverable, not a failure to hide.

---

## 3. Notation

| Symbol | Meaning |
| --- | --- |
| \(t\) | index on the analysis grid (1 minute) |
| \(y_t \in \mathbb{R}^5\) | value channels: `Reservoirs`, `TP3`, `H1`, `DV_pressure`, `Oil_temperature` |
| \(s_t\) | confirmed operating state at \(t\), from §4.3 |
| \(r_t\) | run length in minutes since the last confirmed state transition |
| \(c_t\) | context cell at \(t\) (§6.1) |
| \(a_t\) | Score A statistic (value deviation), §5 |
| \(b_t\) | Score B statistic (state persistence), §6 |
| \(p^A_t, p^B_t\) | calibrated p-values |
| \(B\) | alarm budget, alarms per observed day |
| \(\alpha\) | alarm threshold on the fused p-value |
| \(\delta\) | debounce interval |

---

## 4. Data layer

### 4.1 Analysis grid

Native sampling is 10 s (88.2% of intervals), with jitter 8–13 s and 331 gaps > 60 s.
We aggregate to a **1-minute grid** using the mean of the observations falling in
each minute.

Rules:

- A minute with **no** observation is *missing*, not zero.
- Forward-fill is permitted only across gaps **≤ 2 minutes** (jitter tolerance).
- Any gap **> 2 minutes** breaks the series: no interpolation, no forward-fill, and
  all rolling buffers and the run length are **reset**.
- After a break **> 1 hour**, the first **30 minutes** are additionally discarded as
  warm-up (sensor and model re-stabilisation).
- This reduces 1,516,948 rows to ≈ 253k observed grid points and buys 6× compute
  headroom.

**Why:** the native series is not a uniform grid, so any rolling statistic computed
on row indices silently mixes time scales. Aggregation makes "1 step = 1 minute"
true.

**Failure mode:** averaging can smear a 1–2 minute transient. Accepted, because all
four labelled events last ≥ 4.5 h.

### 4.2 Mandatory masks

| Mask | Definition | Verified (Day 2, recomputed) |
| --- | --- | --- |
| **M1 frozen** | maximal runs where **all 7 analog channels** are simultaneously constant for ≥ 360 consecutive native rows (≥ 1 h) | 7 blocks, 50,425 rows = **3.32%** of native rows, max **51.43 h** |
| **M2 major gap** | intervals > 1 h between consecutive timestamps | 160 intervals, **868.5 h = 17.0%** of wall clock |
| **M3 break warm-up** | first 30 min after any M1 block or M2 gap | 210 rows after frozen blocks + 4,800 after gaps = **5,010** |
| **M4 transition** | rows whose **confirmed** state is `other`, excluded from calibration, scored via back-off | 2,012 rows = **0.66%** of the grid |

Masked rows are excluded from: model fitting, calibration, and the false-alarm
numerator **and denominator**. They are never imputed.

Verified overlap with labels (required check, M1/M2/M3 reported separately):

| Event | Window | M1 | M2 | M3 | Combined | Share |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.0 h | 19 min | 0 | 30 min | 49 min | **3.40%** |
| 2 | 6.52 h | 0 | 0 | 0 | 0 | **0%** |
| 3 | 52.52 h | 0 | 238 min | 30 min | 268 min | **8.51%** |
| 4 | 4.52 h | 0 | 0 | 0 | 0 | **0%** |

- Event 1's 19-minute M1 overlap is a **Day 2 correction, not a new mask**. Day 0
  computed a frozen block's end as `rows × 10 s` from its start. The block spanning
  2020-04-17 09:20:33 → 2020-04-18 00:18:07 contains internal gaps, so its real
  timestamp end is 00:18:07, not the arithmetic 21:45:33. The true tail crosses
  midnight and lands inside event 1. The 30-minute M3 warm-up follows it.
- Event 3's M2 is **two** gaps, not one: the fully-inside 3.784 h gap (2020-06-06
  15:55:17 → 19:42:19) plus the first 0.189 h of the 21.47 h gap that starts at
  2020-06-07 14:19:39, just before the window ends. Day 1 counted only the first,
  reporting 7.2%; the stricter Day 2 value is 238 min = **7.55%**, combined
  **8.51%**. This is the binding number and it remains below the 10% stop
  threshold, with a 1.49-point margin.
- Exposure accounting, stated separately to avoid conflation:
  **M1 removes 3.32% of rows**; **M2 accounts for 868.5 h = 17.0% of the 213-day
  wall clock**; **combined masks cover 21.75% of wall clock**, leaving 77.59% of
  observed rows usable. This is a property of the machine's duty schedule, not a
  defect. The 10% fallback threshold in `experimental_protocol.md` §8 (F6) applies
  **per event window**, not globally.

**Leakage note (corrected Day 2).** The masks are rule-based and label-free, so
they are **not** fitted on the training segment. They are computed once from the
sensor frame and applied unchanged to every split; L4 is therefore implemented as
"the mask is a pure function of the sensor frame", verified by hashing the mask
under two different label sets. There is no mask statistic that could leak.

### 4.3 Operating state (verified encoding)

Documented by the dataset PDF and confirmed empirically on Day 0:

```
raw_state(t) =
    "off"        if Motor_current < 0.5
    "start"      if Motor_current >= 8
    "loaded"     if 5 <= Motor_current < 8 and DV_eletric == 1
    "offloaded"  if 2 <= Motor_current < 5 and DV_eletric == 0
    "other"      otherwise        # ~6.6% of grid rows before hysteresis; transitions and ambiguous rows
```

**Hysteresis (new on Day 1).** Raw state fragments on single-row blips: 143 *raw*
loaded episodes intersect event 3's window despite a 99% loaded share. A candidate
state must therefore persist for **L = 3 consecutive minutes** before it is
confirmed; until then the previous confirmed state is retained:

```
s_t = candidate   if candidate has persisted for >= L minutes
      s_{t-1}     otherwise
```

`start` (44 rows in 7 months) is absorbed into `loaded`, since it is too rare to
calibrate and physically precedes loaded operation.

Result: 4 states — `off`, `offloaded`, `loaded`, `other` — with confirmed shares
**off 50.00%, offloaded 26.17%, loaded 5.52%, other 0.66%** of grid rows.
Post-hysteresis median run lengths are **off 14 min, offloaded 7 min, loaded 4 min,
other 4 min**; median duty cycle is therefore ~14 min off / 7 min offloaded /
4 min loaded. Hysteresis lengthens the shortest confirmed runs and shortens
nothing.

**Calibration cells.** The context cell is `(s_{t-1}, day_type)`, giving six
declared cells: `{off, offloaded, loaded} × {weekday, weekend}`. Rows whose
**lagged** state is `other` are excluded from calibration, so the realized set is
exactly those six cells in every fold. Cell density is a Day 3 risk, not a Day 2
failure: fold 1 is dense (minimum 701 rows), folds 2–3 are adequate except
`loaded|weekend` (52 and 92 rows), and **fold 4's loaded cells are 48 weekday + 29
weekend = 77 rows total**. Both fine cells are below the 50-row refusal floor, so
calibration for `loaded` in fold 4 must back off to the pooled state-level cell
(77 rows), which still clears the 50-row floor but not the 200-row density target.
This is carried forward explicitly.

**Failure mode:** hysteresis delays genuine transitions by up to 3 minutes. This
biases Score B's run length downward by at most 3 minutes, negligible against
episodes of 300–1200 minutes, but it must be stated.

---

## 5. Score A — value anomaly

### 5.1 Definition

For each channel \(c \in \{Reservoirs, TP3, H1, DV\_pressure, Oil\_temperature\}\):

\[
\hat{y}_{c,t} = f_c(x_t), \qquad e_{c,t} = y_{c,t} - \hat{y}_{c,t}, \qquad
a_{c,t} = \frac{|e_{c,t}|}{\hat{\sigma}_{c,t} + \varepsilon}
\]

\[
a_t = \max_{c} \; a_{c,t}
\]

\(x_t\) contains **only information available strictly before \(t\)**:

| Feature group | Contents | Purpose |
| --- | --- | --- |
| Lags | \(y_{c,t-1}, y_{c,t-5}\) | local continuity |
| Rolling level | mean over past 15 min, 1 h, 6 h | drift detection at three horizons |
| Rolling dispersion | MAD over past 1 h | local volatility |
| Calendar | hour, weekday/weekend | tested context arm |
| State | \(s_{t-1}\), \(r_{t-1}\), transitions in past 1 h | tested context arm |

\(\hat{\sigma}_{c,t}\) is a **causal** EWMA of \(|e|\) (half-life 1 h), floored at a
per-channel noise level estimated on training data. Using a causal scale is what
lets the score react to slow faults without a global threshold.

### 5.2 Anti-circularity rules (mandatory)

| Rule | Statement | Rationale |
| --- | --- | --- |
| **R1** | `Motor_current`, `DV_eletric`, `COMP`, `MPG` are **never** Score A targets | They define the state; scoring them with state as context is circular |
| **R2** | State features are lagged \(s_{t-1}\) and hysteresis-confirmed | Same-row state would absorb the fault |
| **R3** | Score A is evaluated in **four context arms**: none / calendar / lagged state / hybrid | The delta between arms *measures* normalization |
| **R4** | `TP2` is excluded from Score A | Day 0: TP2 ≈ 0 in 99.99% of off rows; it is a state indicator, not an independent pressure |
| **R5** | No target channel is used to predict itself contemporaneously | Trivial leakage |

R3 is the operational answer to "how do we prevent regime from normalizing the
fault": we do not assert it is safe, we **measure** the recall loss it causes and
report it. If the lagged-state arm detects fewer events than the no-context arm,
that is a finding, not a bug to hide.

### 5.3 When Score A structurally cannot work

Score A will miss a fault when the fault's only manifestation is *how long* a state
lasts while values remain in their normal conditional range. Event 1 is exactly
this: during its window `Reservoirs` differs from the prior week by only
−0.17 bar. A value residual conditioned on "loaded" sees nothing unusual. This is
the design reason Score B exists.

---

## 6. Score B — state persistence anomaly

### 6.1 Statistic

For confirmed state \(s\), let \(\mathcal{D}_s\) be the **completed** run lengths
(minutes) observed in the calibration segment. At time \(t\) with current run
length \(r_t\):

\[
b_t = \frac{1 + \left|\{d \in \mathcal{D}_{s_t} : d \ge r_t\}\right|}{|\mathcal{D}_{s_t}| + 1}
\]

This is the fraction of historical runs at least as long as the current one. Small
\(b_t\) = unusually persistent state.

### 6.2 Why \(b_t\) is a *score*, not a p-value

At an arbitrary time you are more likely to be inside a long run than a short one
(length-biased sampling). So \(b_t\) is **not uniform** under normal operation, and
must not be read as a false-alarm probability. It is a nonconformity score that is
calibrated in §7 like any other.

This distinction is the most common error in "conformal anomaly detection" write-ups
and is stated explicitly to avoid it.

### 6.3 Alternatives evaluated (ablation, not primary)

| Variant | Definition | Why not primary |
| --- | --- | --- |
| B1 (primary) | empirical survival of run length, above | simple, monotone, one parameter-free statistic |
| B2 | deviation of the last 5 transition periods from their historical distribution | noisier; needs a period estimator; fails on missing cycles |
| B3 | discrete-time hazard: \(P(\text{exit} \mid r)\) from calibration | equivalent ranking to B1 for a single state, more parameters |
| B4 | run length **and** pressure outcome ("demand not met") | risks duplicating Score A; tested as a fusion variant instead |

### 6.4 Known contamination of the reference distribution

Day 0 found **49 of 77 loaded episodes ≥ 30 min lie outside labelled windows**, and
**13 exceed 2 h**, e.g. 796 min on 2020-03-28 and 712 min on 2020-04-12. Some look
benign (`Reservoirs` **rises** +0.44 bar), others look like unlabelled leaks
(2020-05-19, −0.98 bar, 99.5 min).

Consequences, stated as limitations rather than fixed:

1. \(\mathcal{D}_{loaded}\) contains plausible unlabelled faults → thresholds are
   inflated → **recall estimates are conservative**.
2. Any alarm on those episodes counts as a false alarm in our metrics even when it
   may be correct → **precision estimates are pessimistic**.
3. The clean 47-day tail (after event 4) is the only defensible false-alarm
   measurement period, and it too may contain unlabelled faults.

We do not relabel the data. We report both the strict (all-outside-label = false
alarm) and a sensitivity view that excludes the 13 long episodes from the
false-alarm count.

---

## 7. Calibration

### 7.1 Context cells

\[
c_t = (\text{lagged state at } t, \text{ day type})
\]

- Primary granularity: 3 states × 2 day types = **6 cells**, all dense
  (smallest ≈ 74k rows before masking).
- Secondary granularity adds hour: 3 × 2 × 24 = **144 cells**; Day 0 showed 24% of
  cells fall below 200 rows at this resolution, which is why back-off exists.
- Calendar-only (hour × day type) is retained as a **null-control arm**: η² = 0.0017
  for `Reservoirs`, so we expect no benefit and will report that.

### 7.2 Mondrian conformal p-values with back-off

For score \(a\) in cell \(g\) with calibration scores \(S_g\):

\[
p_t = \frac{1 + \left|\{s \in S_g : s \ge a_t\}\right|}{|S_g| + 1}
\]

If \(|S_g| < n_{\min} = 200\), back off to the parent cell:

```
(state, daytype, hour) → (state, daytype) → (state) → global
```

Every alarm reports its cell, \(n_g\), and back-off depth. Cells with \(n_g < 50\)
are never used even after back-off; the alarm reports `insufficient_calibration`.

### 7.3 Validity caveat (must appear in README)

Split-conformal validity requires **exchangeability** within a cell. A time series
with a duty cycle, nightly outages, and seasonal drift is not exchangeable.
Therefore:

> We claim *approximate, empirically verified* conditional false-alarm control,
> not a finite-sample guarantee.

Verification is the per-cell empirical false-alarm rate in
`experimental_protocol.md` §6.7 — if it deviates badly, the claim is withdrawn.

---

## 8. Alarm layer

### 8.1 Fusion

Two calibrated p-values per time step. Because the budget fixes the alarm count,
**fusion is a ranking problem, not a threshold problem**: any monotone rescaling of
the fused statistic is absorbed by the threshold. We therefore compare *rankings*:

| Rule | Statistic | Character |
| --- | --- | --- |
| **OR (primary)** | \(u_t = \min(p^A_t, p^B_t)\) | either channel suffices; best recall, most false alarms per budget |
| AND | \(u_t = \max(p^A_t, p^B_t)\) | both must agree; high precision, misses single-channel faults |
| Weighted | \(u_t = \lambda \tilde{p}^A_t + (1-\lambda)\tilde{p}^B_t\), \(\lambda \in \{0,0.25,0.5,0.75,1\}\) fitted on calibration only | compromise; \(\lambda\) is a hyperparameter and must not be tuned on test |

Note: Bonferroni correction on the OR rule is unnecessary here — it multiplies
\(u_t\) by a constant, which the budget solver absorbs. It would matter only if we
reported a p-value as a significance statement, which we do not.

### 8.2 Brighton/debounce

Raw per-minute alarms are correlated and would blow the budget with a single
episode. Emit an alarm at \(t\) only if:

1. \(u_t \le \alpha\), and
2. no alarm was emitted in \((t-\delta, t]\), with \(\delta = 30\) min.

\(\delta\) is a policy parameter, not learned. It defines "one incident" and is
reported with every result. An alarm carries the **first** triggering timestamp and
the arg-min channel for attribution.

### 8.3 Budget solver

Given target \(B\) alarms/day and calibration segment with observed duration
\(T_{cal}\) days:

\[
\alpha^* = \arg\min_{\alpha \in (0,1]} \left| \frac{N_{alarms}(\alpha; \text{calib})}{T_{cal}} - B \right|
\]

solved by bisection over 40 grid points. If the minimum achievable rate exceeds \(B\)
(e.g. because every episode triggers), report `budget_unreachable` with the floor
rather than silently missing the target.

**Verified, not assumed:** realized rate on the test segment is compared to \(B\).
This is the central empirical claim of the project.

### 8.4 Attribution

Each alarm reports:

| Field | Meaning |
| --- | --- |
| `fired_by` | `A` or `B` (smaller p-value) |
| `p_A`, `p_B` | calibrated p-values |
| `cell`, `n_cell`, `backoff_depth` | calibration provenance |
| `channel` | worst Score A channel, if `fired_by == A` |
| `state`, `run_length_min` | state and persistence at alarm time |
| `context_effect` | \(|f(\text{state}) - f(\text{reference state})| / \hat\sigma\) for Score A |
| `value_effect` | \(a_t\) |

"Which score fired" is what makes the two-score design auditable rather than a
black-box ensemble.

---

## 9. Information overlap — how we prove the scores differ

Two scores can be redundant in three ways. Each has a dedicated test:

| Redundancy type | Test | Threshold for "redundant" |
| --- | --- | --- |
| Statistical | Spearman \(\rho(a_t, b_t)\) on calibration | \(\rho > 0.8\) |
| Rank | Jaccard overlap of top-\(k\) alarm sets, \(k = 20\) | \(> 0.6\) |
| Operational | events detected by **exactly one** score | 0 unique detections |

The operational test is decisive: if neither score detects an event the other
misses, the second score is not earning its place. Expected (not assumed) pattern
from Day 0, to be confirmed:

| Event | Expected primary channel | Reasoning |
| --- | --- | --- |
| 1 | B | no pressure sag (−0.17 bar), longest loaded run (1168 min) |
| 2 | A + B | 308 min run, −0.38 bar sag |
| 3 | A + B | −0.96 bar sag, cycling collapses |
| 4 | A + B | −0.81 bar sag, 153 min run |

"Expected" here is a hypothesis from Day 0 descriptives, **not a result**. If the
table comes out empty for B, the two-score architecture is falsified.

---

## 10. Limitations (stated up front)

1. **Four events.** No significance testing, no AUC-difference claims, no
   generalisation beyond this compressor. Exact counts and wide intervals only.
2. **Exchangeability is false.** Calibration is approximate and must be verified.
3. **Unlabelled anomalies.** 13 long loaded episodes outside labels; recall is
   conservative, precision pessimistic.
4. **Label coarseness.** Windows are maintenance reports, not onsets; detection
   delay is only meaningful relative to window start.
5. **Regime depends on sensor health.** If `Motor_current` fails, Score B degrades
   silently. Not addressed in this scope.
6. **Single machine, single year, COVID period** (Feb–Aug 2020). No seasonal
   generalisation claim.
7. **Calendar context is null on this dataset.** The project cannot claim
   calendar-context benefits from this evidence; that would require another dataset.

---

## 11. Design decision log

| # | Decision | Alternative rejected | Why |
| --- | --- | --- | --- |
| D1 | 1-minute grid | native 10 s | uniform time axis; 6× cheaper; events last hours |
| D2 | Two scores | single CCCS score | Day 0 circularity; fault is a state change |
| D3 | Lagged state as context | same-row state | same-row absorbs the fault |
| D4 | Survival p-value for B | hazard model | fewer parameters, same ranking |
| D5 | Budget-solved threshold | fixed contamination | budget is the operational quantity |
| D6 | OR fusion primary | weighted sum | attribution stays clean; λ needs tuning data |
| D7 | Debounce 30 min | per-row alarms | alarms must map to incidents |
| D8 | Event-level metrics | point PR-AUC | 1.71% positive, 4 events; point metrics mislead |
| D9 | Report null calendar result | drop calendar silently | negative result is part of the evidence |

---

## 12. Day 2 revision notes

| What changed | Day 1 / Day 0 value | Day 2 recomputed value | Why |
| --- | --- | --- | --- |
| Frozen-block end timestamp | arithmetic `start + rows × 10 s` | real last-sample timestamp | internal gaps make arithmetic ends too early; block 1 extends into event 1 |
| Event 1 mask overlap | 0% | M1 19 min + M3 30 min = 3.40% | follows from the corrected block end |
| Event 3 gap overlap | 3.78 h = 7.2% | 238 min = 7.55%, combined 8.51% | a second > 1 h gap starts inside the window |
| M4 calibration filter | current-state `other` removed | **lagged** state `other` removed too | a row after a transition can have a resolved state but an uncallable lagged cell |
| Confirmed `other` share | stated as 1.32% raw | 0.66% of grid rows | hysteresis removes single-row blips; raw `other` is 6.6% of grid rows |
| Mask leakage rule | "fit on train, apply elsewhere" | pure function of the sensor frame | masks use no labels and no fitted statistics |
