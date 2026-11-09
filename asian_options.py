
import numpy as np
from scipy.stats import norm


def asian_call(
    S0, K, T, r, sigma,
    n_simulations=100_000,
    n_steps=12,
    antithetic=False,
    seed=None
):
    """
    Price a discretely monitored arithmetic Asian call.

    Monitoring occurs at t_j = j*T/n_steps, j=1,...,n_steps.

    n_simulations counts complete simulated paths.
    When antithetic=True, it must be even.

    Returns:
        price, standard_error, 95% confidence interval
    """
    if not np.isfinite([S0, K, T, r, sigma]).all():
        raise ValueError("Parameters must be finite")

    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if (
        isinstance(n_simulations, (bool, np.bool_))
        or not isinstance(n_simulations, (int, np.integer))
        or n_simulations < 2
    ):
        raise ValueError("n_simulations must be an integer >= 2")

    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
        or n_steps < 1
    ):
        raise ValueError("n_steps must be a positive integer")

    if antithetic and (n_simulations < 4 or n_simulations % 2):
        raise ValueError(
            "Antithetic sampling requires an even count >= 4"
        )

    if T == 0:
        price = max(S0 - K, 0.0)
        return float(price), 0.0, (float(price), float(price))

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    if sigma == 0:
        times = dt * np.arange(1, n_steps + 1)
        average = np.mean(S0 * np.exp(r * times))
        price = discount * max(average - K, 0.0)
        return float(price), 0.0, (float(price), float(price))

    rng = np.random.default_rng(seed)

    if antithetic:
        n_independent = n_simulations // 2
        prices_plus = np.full(n_independent, S0, dtype=float)
        prices_minus = np.full(n_independent, S0, dtype=float)
        sums_plus = np.zeros(n_independent)
        sums_minus = np.zeros(n_independent)

        for _ in range(n_steps):
            z = rng.standard_normal(n_independent)

            prices_plus *= np.exp(drift + diffusion * z)
            prices_minus *= np.exp(drift - diffusion * z)

            sums_plus += prices_plus
            sums_minus += prices_minus

        payoff_plus = discount * np.maximum(
            sums_plus / n_steps - K, 0.0
        )
        payoff_minus = discount * np.maximum(
            sums_minus / n_steps - K, 0.0
        )

        # Each pair is one independent observation.
        observations = (payoff_plus + payoff_minus) / 2

    else:
        prices = np.full(n_simulations, S0, dtype=float)
        price_sums = np.zeros(n_simulations)

        for _ in range(n_steps):
            z = rng.standard_normal(n_simulations)
            prices *= np.exp(drift + diffusion * z)
            price_sums += prices

        observations = discount * np.maximum(
            price_sums / n_steps - K, 0.0
        )

    price = float(np.mean(observations))
    se = float(
        np.std(observations, ddof=1)
        / np.sqrt(len(observations))
    )

    critical_value = norm.ppf(0.975)
    ci = (
        price - critical_value * se,
        price + critical_value * se
    )

    return price, se, ci
