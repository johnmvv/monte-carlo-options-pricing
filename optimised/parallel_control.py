
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def parallel_moments(S0, K, T, r, sigma, beta, normals):
    """Return sums of adjusted payoffs and squared payoffs."""
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    total = 0.0
    total_sq = 0.0

    for i in prange(len(normals)):
        ST = S0 * np.exp(drift + diffusion * normals[i])
        discounted_stock = discount * ST

        payoff = discount * max(ST - K, 0.0)
        adjusted = payoff - beta * (discounted_stock - S0)

        total += adjusted
        total_sq += adjusted * adjusted

    return total, total_sq


def parallel_control_call(
    S0, K, T, r, sigma, beta,
    n_simulations=1_000_000,
    chunk_size=1_000_000,
    seed=42
):
    """Price using bounded-memory, parallel Monte Carlo."""
    if n_simulations < 2 or chunk_size < 1:
        raise ValueError("Invalid simulation count or chunk size")

    rng = np.random.default_rng(seed)

    total = 0.0
    total_sq = 0.0
    processed = 0

    while processed < n_simulations:
        count = min(chunk_size, n_simulations - processed)

        normals = rng.standard_normal(count)

        batch_total, batch_sq = parallel_moments(
            S0, K, T, r, sigma, beta, normals
        )

        total += batch_total
        total_sq += batch_sq
        processed += count

    mean = total / n_simulations
    variance = max(
        (total_sq - n_simulations * mean**2)
        / (n_simulations - 1),
        0.0
    )

    return mean, np.sqrt(variance / n_simulations)
