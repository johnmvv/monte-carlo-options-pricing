
import numpy as np
from numba import njit


@njit(cache=True)
def asian_payoffs_numba(
    S0, K, T, r, sigma, normals
):
    """Return discounted arithmetic Asian call payoffs.

    normals has shape (n_simulations, n_steps).
    """
    n_simulations, n_steps = normals.shape

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    payoffs = np.empty(n_simulations)

    for i in range(n_simulations):
        price = S0
        total = 0.0

        for j in range(n_steps):
            price *= np.exp(
                drift + diffusion * normals[i, j]
            )
            total += price

        average = total / n_steps
        payoffs[i] = discount * max(average - K, 0.0)

    return payoffs


def asian_payoffs_numpy(
    S0, K, T, r, sigma, normals
):
    """Vectorised NumPy implementation using the same inputs."""
    n_simulations, n_steps = normals.shape

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    prices = np.full(n_simulations, S0, dtype=float)
    totals = np.zeros(n_simulations)

    for j in range(n_steps):
        prices *= np.exp(
            drift + diffusion * normals[:, j]
        )
        totals += prices

    return discount * np.maximum(
        totals / n_steps - K, 0.0
    )
