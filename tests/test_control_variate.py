
import pytest

from black_scholes import black_scholes_call
from control_variate import control_variate_call


def test_reproducibility():
    args = (100, 100, 1, 0.05, 0.2)

    a = control_variate_call(*args, seed=42)
    b = control_variate_call(*args, seed=42)

    assert a == b


def test_at_expiry():
    price, se, ci, beta = control_variate_call(
        110, 100, 0, 0.05, 0.2
    )

    assert price == 10
    assert se == 0
    assert ci == (10, 10)


def test_zero_volatility():
    price, se, _, _ = control_variate_call(
        100, 100, 1, 0.05, 0
    )

    exact = black_scholes_call(100, 100, 1, 0.05, 0)

    assert price == pytest.approx(exact)
    assert se == 0


def test_invalid_parameters():
    with pytest.raises(ValueError):
        control_variate_call(100, 100, -1, 0.05, 0.2)

    with pytest.raises(ValueError):
        control_variate_call(
            100, 100, 1, 0.05, 0.2,
            pilot_simulations=1
        )


def test_analytical_agreement():
    exact = black_scholes_call(100, 100, 1, 0.05, 0.2)

    price, se, _, _ = control_variate_call(
        100, 100, 1, 0.05, 0.2,
        seed=42
    )

    assert abs(price - exact) < 5 * se
