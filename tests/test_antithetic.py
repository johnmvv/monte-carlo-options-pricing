
import pytest

from antithetic import antithetic_call
from black_scholes import black_scholes_call


def test_reproducibility():
    args = (100, 100, 1, 0.05, 0.2)

    result1 = antithetic_call(*args, 10_000, seed=42)
    result2 = antithetic_call(*args, 10_000, seed=42)

    assert result1 == result2


def test_zero_volatility():
    price, se, ci = antithetic_call(
        100, 100, 1, 0.05, 0,
        n_simulations=1000
    )

    exact = black_scholes_call(100, 100, 1, 0.05, 0)

    assert price == pytest.approx(exact)
    assert se == 0
    assert ci == (price, price)


def test_invalid_simulation_count():
    with pytest.raises(ValueError):
        antithetic_call(100, 100, 1, 0.05, 0.2, 999)


def test_analytical_agreement():
    exact = black_scholes_call(100, 100, 1, 0.05, 0.2)

    price, se, _ = antithetic_call(
        100, 100, 1, 0.05, 0.2,
        n_simulations=100_000,
        seed=42
    )

    assert abs(price - exact) < 5 * se
