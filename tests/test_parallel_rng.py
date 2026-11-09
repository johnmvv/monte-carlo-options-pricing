
import numpy as np
import pytest

from black_scholes import black_scholes_call
from optimised.parallel_rng import parallel_rng_call


PARAMS = (100, 100, 1, 0.05, 0.2)


@pytest.mark.parametrize("workers", [2, 4, 6, 12])
def test_accuracy(workers):
    exact = black_scholes_call(*PARAMS)

    price, se = parallel_rng_call(
        *PARAMS,
        n_simulations=120_000,
        n_workers=workers,
        seed=42
    )

    assert abs(price - exact) < 5 * se


def test_reproducibility():
    a = parallel_rng_call(
        *PARAMS, n_simulations=100_000, n_workers=4, seed=42
    )
    b = parallel_rng_call(
        *PARAMS, n_simulations=100_000, n_workers=4, seed=42
    )

    assert a == pytest.approx(b, rel=1e-12)


def test_invalid_workers():
    with pytest.raises(ValueError):
        parallel_rng_call(
            *PARAMS,
            n_simulations=100,
            n_workers=101
        )
