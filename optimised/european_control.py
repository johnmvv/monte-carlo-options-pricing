
from functools import lru_cache

import numpy as np
from numba import njit
from scipy.stats import norm

from black_scholes import black_scholes_call


@lru_cache(maxsize=256)
def optimal_beta(S0, K, T, r, sigma):
    """Population-optimal control coefficient for a European call."""
    params = np.array([S0, K, T, r, sigma], dtype=float)

    if not np.isfinite(params).all():
        raise ValueError("Parameters must be finite")
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if T == 0 or sigma == 0:
        return 0.0

    volatility = sigma * np.sqrt(T)
    variance_parameter = sigma**2 * T

    d1 = (
        np.log(S0 / K) + (r + 0.5 * sigma**2) * T
    ) / volatility

    exact = black_scholes_call(S0, K, T, r, sigma)

    covariance = (
        S0**2 * np.exp(variance_parameter)
        * norm.cdf(d1 + volatility)
        - K * S0 * np.exp(-r * T) * norm.cdf(d1)
        - exact * S0
    )

    variance = S0**2 * np.expm1(variance_parameter)

    return float(covariance / variance)



@njit(cache=True)
def _pricing_kernel(S0, K, T, r, sigma, beta, n, seed):
    """Compute the control-variate price using accumulated moments."""
    if seed >= 0:
        np.random.seed(seed)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    total = 0.0
    total_sq = 0.0

    for _ in range(n):
        z = np.random.standard_normal()
        ST = S0 * np.exp(drift + diffusion * z)

        payoff = discount * max(ST - K, 0.0)
        control = discount * ST

        adjusted = payoff - beta * (control - S0)

        total += adjusted
        total_sq += adjusted * adjusted

    mean = total / n

    sample_variance = max(
        (total_sq - n * mean * mean) / (n - 1),
        0.0
    )

    standard_error = np.sqrt(sample_variance / n)

    return mean, standard_error


@njit(cache=True)
def _pricing_kernel_from_normals(
    S0, K, T, r, sigma, beta, normals
):
    """Calculate control-adjusted payoffs from pre-generated normals."""
    n = len(normals)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    total = 0.0
    total_sq = 0.0

    for i in range(n):
        ST = S0 * np.exp(drift + diffusion * normals[i])

        payoff = discount * max(ST - K, 0.0)
        control = discount * ST

        adjusted = payoff - beta * (control - S0)

        total += adjusted
        total_sq += adjusted * adjusted

    mean = total / n
    variance = max(
        (total_sq - n * mean * mean) / (n - 1),
        0.0
    )

    return mean, np.sqrt(variance / n)



def optimised_control_call(
    S0, K, T, r, sigma,
    n_simulations=100_000,
    seed=None
):
    """European call pricing with an analytical control variate.

    Returns:
        price, standard_error, confidence_interval, beta
    """
    params = np.array([S0, K, T, r, sigma], dtype=float)

    if not np.isfinite(params).all():
        raise ValueError("Parameters must be finite")
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if (
        isinstance(n_simulations, (bool, np.bool_))
        or not isinstance(n_simulations, (int, np.integer))
        or n_simulations < 2
    ):
        raise ValueError("n_simulations must be an integer >= 2")

    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32)")

    if T == 0 or sigma == 0:
        ST = S0 * np.exp(r * T)
        price = float(np.exp(-r * T) * max(ST - K, 0.0))
        return price, 0.0, (price, price), 0.0

    beta = optimal_beta(S0, K, T, r, sigma)

    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_simulations)

    price, se = _pricing_kernel_from_normals(
        S0, K, T, r, sigma, beta, normals
    )

    critical = norm.ppf(0.975)

    ci = (
        float(price - critical * se),
        float(price + critical * se)
    )

    return float(price), float(se), ci, beta
