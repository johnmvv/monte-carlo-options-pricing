
import pytest
from black_scholes import black_scholes_call


def test_known_price():
    price = black_scholes_call(100, 100, 1, 0.05, 0.2)
    assert price == pytest.approx(10.45058357, abs=1e-6)


def test_at_expiry():
    assert black_scholes_call(110, 100, 0, 0.05, 0.2) == 10
    assert black_scholes_call(90, 100, 0, 0.05, 0.2) == 0


def test_zero_volatility():
    price = black_scholes_call(100, 100, 1, 0.05, 0)
    expected = 100 - 100 * 2.718281828459045**(-0.05)
    assert price == pytest.approx(expected)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        black_scholes_call(-100, 100, 1, 0.05, 0.2)

    with pytest.raises(ValueError):
        black_scholes_call(100, 100, -1, 0.05, 0.2)
