
import numpy as np
import pytest

from black_scholes import black_scholes_call
from optimised.european_control import (
    optimal_beta,
    optimised_control_call,
)


def test_optimal_beta_matches_numerical_estimate():
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    rng = np.random.default_rng(42)
    z = rng.standard_normal(1_000_000)

    ST = S0 * np.exp(
        (r - sigma**2 / 2) * T + sigma * np.sqrt(T) * z
    )

    X = np.exp(-r * T) * np.maximum(ST - K, 0)
    Y = np.exp(-r * T) * ST

    empirical_beta = np.cov(X, Y)[0, 1] / np.var(Y, ddof=1)

    assert optimal_beta(S0, K, T, r, sigma) == pytest.approx(
        empirical_beta, abs=0.01
    )


def test_pricing_accuracy():
    params = (100, 100, 1, 0.05, 0.2)
    exact = black_scholes_call(*params)

    price, se, _, _ = optimised_control_call(
        *params, n_simulations=100_000, seed=42
    )

    assert abs(price - exact) < 5 * se


def test_reproducibility():
    params = (100, 100, 1, 0.05, 0.2)

    assert optimised_control_call(*params, seed=42) == (
        optimised_control_call(*params, seed=42)
    )


def test_zero_volatility():
    result = optimised_control_call(100, 100, 1, 0.05, 0)

    assert result[0] == pytest.approx(
        black_scholes_call(100, 100, 1, 0.05, 0)
    )
    assert result[1] == 0


def test_invalid_parameters():
    with pytest.raises(ValueError):
        optimal_beta(100, 100, 1, 0.05, -0.2)

    with pytest.raises(ValueError):
        optimised_control_call(
            100, 100, 1, 0.05, 0.2,
            n_simulations=1
        )


@pytest.mark.parametrize(
    "S0,K,T,r,sigma",
    [
        (100, 80, 1, 0.05, 0.2),
        (100, 100, 1, 0.05, 0.2),
        (100, 120, 1, 0.05, 0.2),
        (100, 100, 2, 0.03, 0.4),
    ]
)
def test_accuracy_across_parameters(S0, K, T, r, sigma):
    params = (S0, K, T, r, sigma)
    exact = black_scholes_call(*params)

    price, se, _, beta = optimised_control_call(
        *params,
        n_simulations=200_000,
        seed=42
    )

    assert np.isfinite(beta)
    assert abs(price - exact) < 5 * se


def test_large_simulation():
    price, se, ci, beta = optimised_control_call(
        100, 100, 1, 0.05, 0.2,
        n_simulations=1_000_000,
        seed=42
    )

    assert np.isfinite(price)
    assert np.isfinite(se)
    assert se > 0
    assert ci[0] < ci[1]


def test_rejects_nonfinite_parameters():
    with pytest.raises(ValueError):
        optimised_control_call(
            100, 100, 1, 0.05, float("nan")
        )
