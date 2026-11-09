
import pytest

from black_scholes import black_scholes_call
from geometric_asian import geometric_asian_call
from asian_control_variate import asian_control_variate_call


def test_one_monitoring_date():
    exact = black_scholes_call(100, 100, 1, 0.05, 0.2)
    result = geometric_asian_call(100, 100, 1, 0.05, 0.2, 1)

    assert result == pytest.approx(exact, rel=1e-12)


def test_at_expiry():
    assert geometric_asian_call(110, 100, 0, 0.05, 0.2) == 10


def test_zero_volatility():
    price = geometric_asian_call(100, 100, 1, 0.05, 0, 12)
    assert price > 0


def test_control_reproducibility():
    a = asian_control_variate_call(
        100, 100, 1, 0.05, 0.2, seed=42
    )
    b = asian_control_variate_call(
        100, 100, 1, 0.05, 0.2, seed=42
    )

    assert a == b


def test_control_at_expiry():
    price, se, ci, beta = asian_control_variate_call(
        110, 100, 0, 0.05, 0.2
    )

    assert price == 10
    assert se == 0
    assert ci == (10, 10)


def test_control_invalid_parameters():
    with pytest.raises(ValueError):
        asian_control_variate_call(
            100, 100, 1, 0.05, 0.2,
            pilot_simulations=1
        )
