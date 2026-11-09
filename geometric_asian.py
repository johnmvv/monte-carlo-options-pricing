
import numpy as np
from scipy.stats import norm


def geometric_asian_call(S0, K, T, r, sigma, n_steps=12):
    """Analytical price of a discretely monitored geometric Asian call.

    Monitoring dates are j*T/n_steps, j=1,...,n_steps.
    The initial price S0 is not included in the average.
    """
    if not np.isfinite([S0, K, T, r, sigma]).all():
        raise ValueError("Parameters must be finite")

    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
        or n_steps < 1
    ):
        raise ValueError("n_steps must be a positive integer")

    if T == 0:
        return float(max(S0 - K, 0.0))

    m = n_steps

    mu = (
        np.log(S0)
        + (r - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    )

    variance = (
        sigma**2 * T * (m + 1) * (2 * m + 1)
        / (6 * m**2)
    )

    discount = np.exp(-r * T)

    if variance == 0:
        return float(
            discount * max(np.exp(mu) - K, 0.0)
        )

    std = np.sqrt(variance)

    d2 = (mu - np.log(K)) / std
    d1 = d2 + std

    price = discount * (
        np.exp(mu + variance / 2) * norm.cdf(d1)
        - K * norm.cdf(d2)
    )

    return float(price)
