
import numpy as np
from scipy.stats import norm


def antithetic_call(S0, K, T, r, sigma, n_simulations=100_000, seed=None):
    """
    Price a European call using antithetic variates.

    n_simulations counts total simulated terminal asset prices.
    It must be an even integer >= 4.

    Returns:
        price: Monte Carlo estimate
        std_error: Standard error based on independent pairs
        ci: Approximate 95% confidence interval
    """
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if (
        isinstance(n_simulations, bool)
        or not isinstance(n_simulations, int)
        or n_simulations < 4
        or n_simulations % 2 != 0
    ):
        raise ValueError("n_simulations must be an even integer >= 4")

    if T == 0:
        payoff = max(S0 - K, 0.0)
        return payoff, 0.0, (payoff, payoff)

    if sigma == 0:
        payoff = np.exp(-r * T) * max(S0 * np.exp(r * T) - K, 0.0)
        return float(payoff), 0.0, (float(payoff), float(payoff))

    rng = np.random.default_rng(seed)
    n_pairs = n_simulations // 2

    Z = rng.standard_normal(n_pairs)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    ST_positive = S0 * np.exp(drift + diffusion * Z)
    ST_negative = S0 * np.exp(drift - diffusion * Z)

    discount = np.exp(-r * T)

    payoff_positive = discount * np.maximum(ST_positive - K, 0.0)
    payoff_negative = discount * np.maximum(ST_negative - K, 0.0)

    pair_payoffs = (payoff_positive + payoff_negative) / 2

    price = np.mean(pair_payoffs)
    std_error = np.std(pair_payoffs, ddof=1) / np.sqrt(n_pairs)

    z = norm.ppf(0.975)
    ci = (price - z * std_error, price + z * std_error)

    return float(price), float(std_error), tuple(map(float, ci))
