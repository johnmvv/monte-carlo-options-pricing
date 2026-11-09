
import pytest

from monte_carlo import monte_carlo_call
from black_scholes import black_scholes_call


def test_reproducibility():
    args = (100, 100, 1, 0.05, 0.2)

    result1 = monte_carlo_call(*args, n_simulations=10_000, seed=42)
    result2 = monte_carlo_call(*args, n_simulations=10_000, seed=42)

    assert result1 == result2


def test_at_expiry():
    price, se, ci = monte_carlo_call(
        110, 100, 0, 0.05, 0.2
    )

    assert price == 10
    assert se == 0
    assert ci == (10, 10)


def test_analytical_agreement():
    exact = black_scholes_call(100, 100, 1, 0.05, 0.2)

    price, se, _ = monte_carlo_call(
        100, 100, 1, 0.05, 0.2,
        n_simulations=100_000,
        seed=42
    )

    assert abs(price - exact) < 5 * se


def test_invalid_parameters():
    with pytest.raises(ValueError):
        monte_carlo_call(100, 100, 1, 0.05, -0.2)

    with pytest.raises(ValueError):
        monte_carlo_call(
            100, 100, 1, 0.05, 0.2,
            n_simulations=1
        )
