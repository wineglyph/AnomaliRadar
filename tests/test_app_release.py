"""Release-level tests: frozen metrics, app loaders, smoke pages and packaging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app_lib import artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_required_release_files_exist() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CITATION.cff",
        ".gitignore",
        "pyproject.toml",
        "docs/release_report.md",
        "app.py",
        "app_lib/artifacts.py",
        "app_lib/ui.py",
        "app_pages/overview.py",
        "app_pages/data_explorer.py",
        "app_pages/anomaly_detection.py",
        "app_pages/alarm_details.py",
        "app_pages/model_comparison.py",
        "app_pages/limitations.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert missing == []


def test_frozen_artifacts_are_present() -> None:
    assert artifacts.missing_artifacts() == []


def test_frozen_identity_matches_summary() -> None:
    summary = artifacts.load_summary()
    manifest = artifacts.load_manifest()
    assert summary["config_hash"] == "06954d34ced38410"
    assert manifest["config_hash"] == summary["config_hash"]
    assert (
        manifest["dataset_fingerprint"]
        == "420986c98d243c8468e233a5557955956cd4e6e8e457285548f779fcd147aaed"
    )
    assert summary["n_falsified"] == 0
    assert all(verdict == "PASS" for verdict in summary["falsification_verdicts"].values())


def test_headline_metrics_match_frozen_csv() -> None:
    table = artifacts.pooled_comparison(1.0)
    or_row = table[table["method"] == "P3_or"].iloc[0]
    assert or_row["recall"] == "2/4"
    assert int(or_row["alarms"]) == 11
    assert int(or_row["false_alarms"]) == 1
    assert or_row["mean_abs_budget_error"] == pytest.approx(1.6444626614872817)
    assert or_row["tail_per_day"] == pytest.approx(0.5609348914858097)
    assert or_row["tail_relative_error"] == pytest.approx(-0.43906510851419034)

    a_row = table[table["method"] == "P1_A_hybrid"].iloc[0]
    b_row = table[table["method"] == "P2_B_B1"].iloc[0]
    assert (a_row["recall"], int(a_row["alarms"]), int(a_row["false_alarms"])) == ("1/4", 2, 0)
    assert (b_row["recall"], int(b_row["alarms"]), int(b_row["false_alarms"])) == ("1/4", 9, 1)


def test_event_level_outcomes_match_report() -> None:
    matrix = artifacts.event_matrix(1.0).set_index("method")
    or_row = matrix.loc["P3_or"]
    assert or_row["E1"] == "missed"
    assert or_row["E2"] == "missed"
    assert or_row["E3"] == "detected"
    assert or_row["E4"] == "detected"
    assert or_row["delay_3"] == pytest.approx(-10.0)
    assert or_row["delay_4"] == pytest.approx(39.0)


def test_tail_is_stepwise_not_proportional() -> None:
    frame = artifacts.load_fold_results()
    or_tail = frame[(frame["fold"] == "T") & (frame["method"] == "P3_or")].set_index("budget")
    assert int(or_tail.loc[0.5, "alarms"]) == int(or_tail.loc[1.0, "alarms"]) == 14
    assert or_tail.loc[1.0, "alarms_per_day"] == pytest.approx(0.5609348914858097)


def test_sparse_fold4_loaded_calibration_is_reported() -> None:
    sizes = artifacts.load_cell_sizes()
    fold4 = sizes[
        (sizes["fold"] == 4)
        & (sizes["score"].isin(["P1_A_hybrid", "P2_B_B1"]))
        & (sizes["level"] == "state")
        & (sizes["key"] == "loaded")
    ].set_index("score")["n"]
    assert int(fold4["P1_A_hybrid"]) == 41
    assert int(fold4["P2_B_B1"]) == 30


def test_alarm_table_contract_is_stable() -> None:
    alarms = artifacts.load_alarms()
    assert len(alarms) == 5899
    expected = {
        "fold",
        "method",
        "budget",
        "timestamp",
        "position",
        "p",
        "alpha",
        "p_a",
        "p_b",
        "fired_by",
        "cell",
        "cal_level",
        "cal_n",
        "cal_depth",
        "channel",
        "state",
        "run_length_min",
        "context_effect",
        "value_effect",
        "event_related",
    }
    assert expected.issubset(set(alarms.columns))
    assert alarms["event_related"].sum() == 1070


def test_download_script_keeps_pinned_checksum() -> None:
    text = (ROOT / "scripts" / "download_data.py").read_text(encoding="utf-8")
    assert "EXPECTED_SHA256" in text
    assert "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a" in text


def test_readme_contains_required_commands_and_limitations() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for command in (
        "python scripts/download_data.py",
        "python scripts/run_experiments.py",
        "python scripts/make_figures.py",
        "streamlit run app.py",
        "python -m pytest -q",
    ):
        assert command in readme
    for phrase in (
        "2 of 4",
        "0.5609",
        "four labelled events",
        "CC BY 4.0",
        "MIT",
    ):
        assert phrase in readme


def test_no_machine_specific_absolute_paths_or_obvious_secrets() -> None:
    suffixes = {".py", ".md", ".toml", ".cff", ".txt", ".yml", ".yaml", ".json", ".csv"}
    excluded = {".venv", "data", ".pytest_cache", "__pycache__", ".git", "dist"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in excluded for part in path.parts):
            continue
        if path.name == "test_app_release.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "/home/" in text or "C:\\Users\\" in text or "Masaüstü" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_streamlit_pages_smoke() -> None:
    app_test = pytest.importorskip("streamlit.testing.v1")
    expected = {
        "app_pages/overview.py": "AnomaliRadar",
        "app_pages/data_explorer.py": "Data Explorer",
        "app_pages/anomaly_detection.py": "Anomaly Detection",
        "app_pages/alarm_details.py": "Alarm Details",
        "app_pages/model_comparison.py": "Model Comparison",
        "app_pages/limitations.py": "Limitations",
    }
    at = app_test.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert len(at.exception) == 0
    for page, title in expected.items():
        at.switch_page(page).run()
        assert len(at.exception) == 0, f"{page}: {[e.value for e in at.exception]}"
        assert any(item.value == title for item in at.title), page
