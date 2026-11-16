# Contributing to AnomaliRadar

Thanks for your interest. This repository is a **frozen research release**: the
scientific implementation, experiment outputs, metrics and F1–F9 verdicts are
locked. Contributions are welcome for documentation, packaging, the Streamlit
viewer, tests and genuine bug fixes. They are not welcome for retuning,
retraining, metric changes or post-hoc result selection.

## Ground rules

1. **Do not change reported results or evaluation rules** to make a metric look
   better. If a metric changes, the change must be a scientifically justified
   correction, documented transparently and accompanied by a full rerun.
2. **Do not refit or retune** `src/anomaliradar/model/` as part of an app or
   documentation change.
3. **Do not commit the dataset.** MetroPT-3 is CC BY 4.0 and is downloaded by
   `scripts/download_data.py`. `data/` is git-ignored.
4. **Preserve the frozen artefacts.** The files in `reports/experiments/` are the
   evidence for the release. Do not hand-edit them. Regenerate them with the
   deterministic scripts if a genuine scientific correction requires it.
5. **Do not add demo or synthetic results** to the application. Missing
   artefacts must be shown as missing, never replaced with invented values.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

The downloader verifies the official UCI archive against the pinned SHA-256 in
`scripts/download_data.py`.

## Tests

Run the complete suite before opening a pull request:

```bash
python -m pytest -q
```

The scientific suite must stay green. If you add a test, keep it deterministic
and independent of the network. Dataset-dependent tests should skip cleanly when
the CSV is absent, following the existing `pytest.mark.skipif` pattern.

## App changes

- Keep the app **read-only** with respect to the experiment.
- Load artefacts through `app_lib/artifacts.py`; do not duplicate paths or
  parsing logic in pages.
- Handle a missing dataset and missing artefacts gracefully.
- Run the app locally:

  ```bash
  streamlit run app.py
  ```

- Add or update a smoke test when a new page or loader is introduced.

## Documentation changes

- Keep numbers consistent with `reports/experiments/`.
- Use “empirical calibration” and “empirical false-alarm control”, never
  “distribution-free guarantee”.
- State the four-event, 7.746-observed-day sample when discussing recall.
- Do not describe the algorithms as newly invented. The contribution is an
  auditable composition.

## Code style

- Python 3.11+.
- Follow the existing module structure and type hints.
- Keep changes small and focused; avoid unrelated reformatting.
- Prefer the existing helpers in `app_lib/ui.py` over new one-off UI code.

## Reporting a bug

Include:

- the command or page that failed;
- the full traceback;
- the config hash and dataset fingerprint from the sidebar or
  `reports/experiments/run_manifest.json`;
- whether `data/raw/metropt3/MetroPT3(AirCompressor).csv` is present;
- the output of `python -m pytest -q` if the failure is scientific.

## Pull requests

A pull request should state:

1. what changed and why;
2. whether any reported metric or frozen artefact changed (normally: no);
3. the test command and result;
4. any new limitation the change introduces.

By contributing, you agree that your code contribution is licensed under the
MIT licence in `LICENSE`. Dataset contributions are not accepted.
