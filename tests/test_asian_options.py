
import pytest

from asian_options import asian_call
from black_scholes import black_scholes_call


def test_reproducibility():
    args = (100, 100, 1, 0.05, 0.2)
    assert asian_call(*args, seed=42) == asian_call(*args, seed=42)


def test_at_expiry():
    price, se, ci = asian_call(110, 100, 0, 0.05, 0.2)
    assert price == 10
    assert se == 0
    assert ci == (10, 10)


def test_zero_volatility():
    price, se, _ = asian_call(
        100, 100, 1, 0.05, 0, n_steps=12
    )

    assert price > 0
    assert se == 0


def test_one_monitoring_date_equals_european():
    exact = black_scholes_call(100, 100, 1, 0.05, 0.2)

    price, se, _ = asian_call(
        100, 100, 1, 0.05, 0.2,
        n_steps=1,
        n_simulations=100_000,
        seed=42
    )

    assert abs(price - exact) < 5 * se


def test_antithetic_reproducibility():
    args = (100, 100, 1, 0.05, 0.2)

    a = asian_call(*args, antithetic=True, seed=42)
    b = asian_call(*args, antithetic=True, seed=42)

    assert a == b


def test_invalid_path_count():
    with pytest.raises(ValueError):
        asian_call(100, 100, 1, 0.05, 0.2, n_simulations=1)

    with pytest.raises(ValueError):
        asian_call(
            100, 100, 1, 0.05, 0.2,
            n_simulations=999,
            antithetic=True
        )


def test_invalid_monitoring_count():
    with pytest.raises(ValueError):
        asian_call(100, 100, 1, 0.05, 0.2, n_steps=0)
