
import pytest

from benchmark import run_method


@pytest.mark.parametrize(
    "method",
    ["Standard", "Antithetic", "Control variate"]
)
def test_benchmark_reproducibility(method):
    first = run_method(method, 1000, seed=42)
    second = run_method(method, 1000, seed=42)

    assert first == second


@pytest.mark.parametrize(
    "method",
    ["Standard", "Antithetic", "Control variate"]
)
def test_benchmark_returns_finite_price(method):
    import numpy as np

    price = run_method(method, 1000, seed=42)

    assert np.isfinite(price)
    assert price >= 0


def test_unknown_method():
    with pytest.raises(ValueError):
        run_method("Unknown", 1000, seed=42)
