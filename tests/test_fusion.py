"""Day 3 fusion tests: OR, AND, weighted, Bonferroni and refusal semantics.

At a fixed budget fusion is a ranking problem. These tests pin the algebra and
the handling of a refused side; they do not claim statistical significance.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomaliradar.model.fusion import FUSION_LAMBDAS, fuse, weighted_grid


def test_or_and_and_weighted_algebra() -> None:
    a = np.array([0.01, 0.20, 0.90])
    b = np.array([0.30, 0.05, 0.80])

    np.testing.assert_allclose(fuse(a, b, rule="or").p, [0.01, 0.05, 0.80])
    np.testing.assert_allclose(fuse(a, b, rule="and").p, [0.30, 0.20, 0.90])
    np.testing.assert_allclose(
        fuse(a, b, rule="weighted", lambda_=0.5).p, [0.155, 0.125, 0.85]
    )
    np.testing.assert_allclose(
        fuse(a, b, rule="weighted", lambda_=1.0).p, a
    )
    np.testing.assert_allclose(
        fuse(a, b, rule="weighted", lambda_=0.0).p, b
    )


def test_or_bonferroni_is_a_two_test_correction() -> None:
    a = np.array([0.01, 0.60, 0.90])
    b = np.array([0.30, 0.70, 0.95])
    corrected = fuse(a, b, rule="or_bonferroni").p
    np.testing.assert_allclose(corrected, [0.02, 1.0, 1.0])
    assert np.all(corrected >= fuse(a, b, rule="or").p)


def test_refused_side_is_handled_explicitly() -> None:
    a = np.array([0.01, 0.01, 0.01])
    b = np.array([0.20, 0.20, 0.20])
    eligible_a = np.array([True, False, False])
    eligible_b = np.array([False, True, False])

    or_result = fuse(
        a, b, rule="or", eligible_a=eligible_a, eligible_b=eligible_b
    )
    # Row 0: only A is available; row 1: only B; row 2: neither -> p=1.
    np.testing.assert_allclose(or_result.p, [0.01, 0.20, 1.0])

    and_result = fuse(
        a, b, rule="and", eligible_a=eligible_a, eligible_b=eligible_b
    )
    np.testing.assert_allclose(and_result.p, [1.0, 1.0, 1.0])

    weighted = fuse(
        a, b, rule="weighted", lambda_=0.5,
        eligible_a=eligible_a, eligible_b=eligible_b,
    )
    # A refused side enters the weighted sum as the least anomalous value 1.
    np.testing.assert_allclose(weighted.p, [0.505, 0.60, 1.0])


def test_nan_inputs_are_treated_as_least_anomalous() -> None:
    a = np.array([np.nan, 0.02])
    b = np.array([0.03, np.nan])
    result = fuse(a, b, rule="or")
    np.testing.assert_allclose(result.p, [0.03, 0.02])


def test_weighted_grid_covers_the_pre_registered_lambdas() -> None:
    a = np.array([0.1, 0.9])
    b = np.array([0.2, 0.8])
    grid = weighted_grid(a, b)
    assert set(grid) == set(FUSION_LAMBDAS)
    assert grid[0.0].lambda_ == 0.0
    assert grid[1.0].lambda_ == 1.0
    with pytest.raises(ValueError):
        fuse(a, b, rule="weighted", lambda_=1.5)
    with pytest.raises(ValueError):
        fuse(a, b, rule="unknown")
