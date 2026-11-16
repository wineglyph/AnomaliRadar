# Release report — AnomaliRadar 0.2.0

**Date:** 2026-11-16
**Release:** AnomaliRadar 0.2.0 (final research-engineering release)
**Config hash:** `06954d34ced38410`
**Dataset fingerprint:** `420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed`
**Scientific verdict:** CONDITIONAL GO; F1–F9 all PASS, `n_falsified=0`

---

## 1. Scope and freeze declaration

This release packages the completed AnomaliRadar project. The following are
frozen and were not changed:

- Score A and Score B implementations.
- Calibration implementation and back-off/refusal rules.
- Alarm fusion and the 30-minute debounce.
- Alarm-budget configuration and the primary 1/day budget.
- Masks M1–M3, chronological folds and the frozen tail reuse.
- All experiment outputs in `reports/experiments/`.
- All reported metrics and F1–F9 verdicts.

No model was retrained, retuned or redesigned. No metric or evaluation rule was
changed. The release adds a read-only Streamlit viewer, documentation, packaging,
tests and a clean source archive.

Two scientifically neutral release changes were made:

1. `scripts/download_data.py` now pins the official archive SHA-256
   `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a` and rejects a
   downloaded archive that does not match it. `--verify-only` checks both the
   pinned checksum and the recorded manifest. The download and extraction
   mechanism is preserved.
2. The package version was aligned to `0.2.0` in `pyproject.toml`,
   `CITATION.cff` and `src/anomaliradar/__init__.py`, matching the existing
   `__version__`.
3. One pre-existing reproducibility test now skips cleanly when the MetroPT-3
   CSV is absent. With the dataset present it still runs and asserts the same
   byte-stable replay as before; the skip only makes a clean clone usable
   without the dataset.

---

## 2. Deliverables

| Deliverable | Location |
| --- | --- |
| Streamlit application | `app.py`, `app_lib/`, `app_pages/` |
| Final README | `README.md` |
| MIT licence | `LICENSE` |
| Contributing guide | `CONTRIBUTING.md` |
| Citation metadata | `CITATION.cff` |
| Dependency and pytest configuration | `pyproject.toml` |
| Updated documentation | `docs/` |
| Release tests | `tests/test_app_release.py` |
| Clean source archive | `dist/anomaliradar-0.2.0-source.tar.gz` |
| Archive checksum | `dist/anomaliradar-0.2.0-source.tar.gz.sha256` |
| This report | `docs/release_report.md` |

### Streamlit pages

| Page | Content |
| --- | --- |
| **Overview** | Project introduction, dataset summary, recommended configuration, headline metrics, limitations, reproduction commands. |
| **Data Explorer** | Sensor time series, operating-state timeline, masked regions, failure windows, data gaps; reads the local CSV only. |
| **Anomaly Detection** | Score A, Score B, OR p-values, calibration-selected alpha, alarm timestamps, calibration diagnostics, sensor effects; optional local snapshot. |
| **Alarm Details** | Timestamp, triggering score, sensor information, state, calibration level and `n`, plain-language explanation, optional sensor snapshot. |
| **Model Comparison** | Baseline comparison, A/B/OR/AND/weighted, event-level recall, alarm rates, delays, calibration diagnostics. |
| **Limitations** | Four-event sample, two OR detections, 0.5609/day tail, no strict 1/day guarantee, sparse loaded calibration, non-exchangeability, dataset-specific conclusions. |

The app reads only frozen artefacts and the local dataset. Missing data is shown
as missing; no demo or synthetic result is substituted.

---

## 3. Verification evidence

The exact commands and results are recorded here. Commands are run from the
repository root with the virtual environment activated.

| Check | Command | Result |
| --- | --- | --- |
| Frozen scientific suite | `python -m pytest -q` (before release tests) | **141 passed in 92.90 s** |
| Complete release suite | `python -m pytest -q` | **153 passed in 126.72 s** |
| Clean-room archive suite | extract archive, then `python -m pytest -q` | **131 passed, 22 skipped, 0 failed in 12.00 s** |
| App page smoke | `pytest tests/test_app_release.py::test_streamlit_pages_smoke` | Six pages, **0 exceptions** |
| Streamlit startup | `streamlit run app.py --server.headless true --server.port 8611` | Server started; `/_stcore/health` returned **ok**, `/` returned **200** |
| Streamlit version | `streamlit --version` | **1.67.0** |
| Import check | Python import of `anomaliradar`, data layer, model layer and `app_lib` | **OK** |
| Dependency install | `pip install -e ".[app,dev]"` | **OK**, editable install built and installed as 0.2.0 |
| Wheel build and metadata | `pip wheel . --no-deps` | **OK**, `License-Expression: MIT`, LICENSE included, all runtime deps declared |
| Dataset verification | `python scripts/download_data.py --verify-only` | **OK**, pinned = recorded = actual |
| Experiment CLI | `python scripts/run_experiments.py --folds 1 --no-ablations --no-baselines --output /tmp/anomaliradar_fold1` | Completed; fold-1 predictions **byte-identical** to the frozen file |
| Fold-1 prediction SHA-256 | frozen vs replay | `9a8bcd6714a258be526590e85231d2b30b354d843a9ced5c6bd4f8bd4e0057db` (both) |
| Figure CLI | `python scripts/make_figures.py --help` | **OK** |
| README commands | `download_data.py --verify-only`, `run_experiments.py --help`, `make_figures.py --help`, `pytest`, `streamlit run` | All present and executable |
| Metric consistency | `tests/test_app_release.py` | Headline metrics, event outcomes, tail step function and fold-4 loaded rows match the CSVs |

The full-suite count includes the original 141 scientific tests plus 12 release
tests: 10 artifact/documentation checks, 1 absolute-path/secret audit and 1
six-page Streamlit smoke test. The original 141 tests were not modified.

The clean-room archive run extracts the release without `data/` or `.venv/`.
Dataset-dependent tests skip cleanly; the app smoke tests still render the
missing-dataset state and the metric pages still pass. A run with the dataset
present reports **153 passed**.

The fold-1 replay is a command-level check, not a new experiment. A single-fold
run cannot reproduce the all-fold falsification verdicts, and it appropriately
reports different F1/F2 outcomes because only one event is present. The frozen
all-fold artefacts remain the authoritative result.

---

## 4. Release audit

### 4.1 Excluded content

The source archive excludes:

- `data/` (including the 218,300,507-byte MetroPT-3 CSV and the 218,381,995-byte
  archive);
- `.venv/` and any virtual environment;
- `.pytest_cache/`, `__pycache__/`, `*.pyc`, `*.pyo`;
- `dist/`, `build/`, `*.egg-info/`, `*.whl` and local logs;
- `*.zip`, `*.part` and temporary files;
- `.git/`, editor state and `.streamlit/secrets.toml`.

The dataset is fetched by `scripts/download_data.py` and verified against the
pinned SHA-256. The source archive preserves that mechanism.

### 4.2 Credentials and machine-specific paths

- No API keys, tokens, private keys, passwords or `.env` files are present.
- The release test `test_no_machine_specific_absolute_paths_or_obvious_secrets`
  scans source, tests, docs, configuration and report files for Unix
  home-directory paths, Windows user paths and the local machine name.
  **No offenders.**
- No absolute path remains in README, docs, app, scripts or tests.
- The only machine-specific value in the release is the historical manifest
  timestamp; it is not a path or credential.

### 4.3 Metric consistency

The following were checked against the actual CSVs, not against the reports:

| Claim | Actual source | Result |
| --- | --- | --- |
| OR recall at 1/day | `fold_results.csv`, `P3_or`, folds 1–4 | 2/4 (events 3 and 4) |
| OR alarms / false alarms at 1/day | same | 11 / 1 |
| OR mean absolute budget error at 1/day | same | 1.6444626614872817 |
| OR tail rate at 1/day | `fold_results.csv`, fold T | 0.5609348914858097/day |
| OR tail relative error | same | −0.43906510851419034 |
| 0.5/day and 1/day tail alarm count | same | 14 and 14 |
| Fold-4 loaded reference rows | `calibration_cell_sizes.csv` | A 41, B 30 |
| Alarm table shape | `alarms.csv` | 5,899 rows × 20 columns |
| Event-related alarms | `alarms.csv` | 1,070 |
| F1–F9 | `summary.json` | 9 PASS, 0 FAIL, `n_falsified=0` |

No mismatch was found.

---

## 5. Archive identity

The clean source archive is created with `tar` from the repository root, using
the exclusion list in §4.1. It contains **138 entries**: source, tests,
documentation, figures and experiment reports, but **not** the dataset, virtual
environment, build output or caches. The MetroPT-3 CSV and archive are absent;
the download-and-verify script is included.

```text
dist/anomaliradar-0.2.0-source.tar.gz
dist/anomaliradar-0.2.0-source.tar.gz.sha256
```

The SHA-256 is recorded in the adjacent `.sha256` file rather than embedded in
this report, so that the report can be included in the archive without making the
archive hash self-referential. Verify with:

```bash
sha256sum -c dist/anomaliradar-0.2.0-source.tar.gz.sha256
tar -tzf dist/anomaliradar-0.2.0-source.tar.gz | head
```

The archive is intentionally not added to Git; `dist/` is git-ignored. The
archive is a release artefact, not source to be committed. It is attached to the
GitHub release `v0.2.0` at <https://github.com/wineglyph/AnomaliRadar/releases/tag/v0.2.0>
together with its `.sha256` record.

---

## 6. Publication status

**Published.**

| Item | Value |
| --- | --- |
| Repository | <https://github.com/wineglyph/AnomaliRadar> |
| Visibility | Public |
| Default branch | `main` |
| Remote | `https://github.com/wineglyph/AnomaliRadar.git` |
| Release tag | `v0.2.0` |
| Release page | <https://github.com/wineglyph/AnomaliRadar/releases/tag/v0.2.0> |
| Source archive asset | `anomaliradar-0.2.0-source.tar.gz` (+ `.sha256`) |
| Force-push | Not used |

The published commit is the one tagged `v0.2.0`. Re-verify with:

```bash
git remote -v
git rev-parse HEAD
git ls-remote --heads origin main
gh repo view wineglyph/AnomaliRadar --json visibility,url,defaultBranchRef
gh release view v0.2.0 --json tagName,isDraft,isPrerelease,assets \
  --repo wineglyph/AnomaliRadar
```

The working tree is a Git repository whose `origin` is the URL above. No other
account and no other repository was used. No GitHub release existed before the
`v0.2.0` release described here.

---

## 7. Known limitations carried into the release

The release does not soften the scientific limitations:

- Four labelled events and 7.746 observed event-fold days.
- OR detects events 3 and 4 only at 1/day; events 1 and 2 are missed.
- Event 3 is detected 10 minutes early; event 4 is detected 39 minutes late by
  OR (Score B alone is 32 minutes late).
- The independent tail realizes 0.5609 alarms/day at a 1/day request (−43.9%).
- The budget response is stepwise; 0.5/day and 1/day select the same 14 tail
  alarms.
- Fold-4 loaded calibration has 41 (A) / 30 (B) reference rows, below the
  50-row refusal threshold; event 4 uses global calibration.
- Event 1's detection-credit window is 35.2% masked.
- The series is not exchangeable; calibration is empirical, not
  distribution-free.
- The conclusion is dataset-specific; high budgets are not certified.

These are documented in `README.md`, `docs/final_evaluation.md` and the app's
Limitations page.

---

## 8. Final state

The project is left in a state where a user can:

1. install the scientific and app dependencies with
   `pip install -e ".[app,dev]"`;
2. download and verify the dataset with `python scripts/download_data.py`;
3. reproduce the experiment with
   `python scripts/run_experiments.py --output reports/experiments`;
4. regenerate figures with `python scripts/make_figures.py`;
5. run the complete suite with `python -m pytest -q`;
6. launch the viewer with `streamlit run app.py`;
7. inspect the clean source archive and its checksum in `dist/` or download them
   from the `v0.2.0` GitHub release;
8. clone and use the published repository at
   <https://github.com/wineglyph/AnomaliRadar>.

**This is the final release task. No further development phase is proposed.**
