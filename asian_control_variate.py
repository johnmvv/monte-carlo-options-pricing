
import numpy as np
from scipy.stats import norm

from geometric_asian import geometric_asian_call


def asian_control_variate_call(
    S0, K, T, r, sigma,
    n_simulations=100_000,
    n_steps=12,
    pilot_simulations=10_000,
    seed=None
):
    """Price an arithmetic Asian call using a geometric control variate.

    Returns:
        price, standard_error, confidence_interval, beta
    """
    if not np.isfinite([S0, K, T, r, sigma]).all():
        raise ValueError("Parameters must be finite")

    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    for n in (n_simulations, pilot_simulations, n_steps):
        if (
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 1
        ):
            raise ValueError("Counts must be positive integers")

    if n_simulations < 2 or pilot_simulations < 2:
        raise ValueError("Simulation counts must be >= 2")

    if T == 0:
        price = float(max(S0 - K, 0.0))
        return price, 0.0, (price, price), 0.0

    if sigma == 0:
        dt = T / n_steps
        times = dt * np.arange(1, n_steps + 1)
        average = np.mean(S0 * np.exp(r * times))
        price = float(np.exp(-r * T) * max(average - K, 0.0))
        return price, 0.0, (price, price), 0.0

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    exact_geometric = geometric_asian_call(
        S0, K, T, r, sigma, n_steps
    )

    def sample(n):
        log_prices = np.full(n, np.log(S0))
        arithmetic_sums = np.zeros(n)
        log_sums = np.zeros(n)

        for _ in range(n_steps):
            z = rng.standard_normal(n)
            log_prices += drift + diffusion * z

            arithmetic_sums += np.exp(log_prices)
            log_sums += log_prices

        arithmetic_average = arithmetic_sums / n_steps
        geometric_average = np.exp(log_sums / n_steps)

        X = discount * np.maximum(
            arithmetic_average - K, 0.0
        )
        Y = discount * np.maximum(
            geometric_average - K, 0.0
        )

        return X, Y

    # Independent pilot sample
    X_pilot, Y_pilot = sample(pilot_simulations)

    covariance = np.cov(X_pilot, Y_pilot, ddof=1)[0, 1]
    variance_y = np.var(Y_pilot, ddof=1)

    beta = covariance / variance_y if variance_y > 0 else 0.0

    # Main pricing sample
    X, Y = sample(n_simulations)

    adjusted = X - beta * (Y - exact_geometric)

    price = float(np.mean(adjusted))
    se = float(
        np.std(adjusted, ddof=1) / np.sqrt(n_simulations)
    )

    z = norm.ppf(0.975)
    ci = (
        float(price - z * se),
        float(price + z * se)
    )

    return price, se, ci, float(beta)
