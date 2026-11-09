
import numpy as np
import pytest

from numba_asian import (
    asian_payoffs_numba,
    asian_payoffs_numpy,
)


PARAMS = (100, 100, 1, 0.05, 0.2)


@pytest.mark.parametrize(
    "n_paths,n_steps",
    [(10, 1), (100, 12), (1000, 52)]
)
def test_numba_matches_numpy(n_paths, n_steps):
    rng = np.random.default_rng(42)
    normals = rng.standard_normal((n_paths, n_steps))

    numpy_result = asian_payoffs_numpy(*PARAMS, normals)
    numba_result = asian_payoffs_numba(*PARAMS, normals)

    np.testing.assert_allclose(
        numba_result,
        numpy_result,
        rtol=1e-12,
        atol=1e-12
    )


def test_zero_volatility():
    normals = np.zeros((100, 12))
    params = (100, 100, 1, 0.05, 0)

    numpy_result = asian_payoffs_numpy(*params, normals)
    numba_result = asian_payoffs_numba(*params, normals)

    np.testing.assert_allclose(
        numba_result,
        numpy_result,
        rtol=1e-12,
        atol=1e-12
    )


def test_reproducibility():
    rng = np.random.default_rng(42)
    normals = rng.standard_normal((1000, 12))

    first = asian_payoffs_numba(*PARAMS, normals)
    second = asian_payoffs_numba(*PARAMS, normals)

    np.testing.assert_array_equal(first, second)


def test_nonnegative_payoffs():
    rng = np.random.default_rng(42)
    normals = rng.standard_normal((1000, 12))

    payoffs = asian_payoffs_numba(*PARAMS, normals)

    assert np.all(payoffs >= 0)
