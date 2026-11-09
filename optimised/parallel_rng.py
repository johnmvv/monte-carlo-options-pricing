
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from optimised.european_control import (
    optimal_beta,
    _pricing_kernel_from_normals,
)


def _worker(args):
    """Generate random samples and price one independent batch."""
    S0, K, T, r, sigma, beta, n, seed_sequence = args

    rng = np.random.default_rng(seed_sequence)
    normals = rng.standard_normal(n)

    price, se = _pricing_kernel_from_normals(
        S0, K, T, r, sigma, beta, normals
    )

    # Convert sample statistics back to mergeable moments.
    sample_variance = (se**2) * n

    return n, price, sample_variance


def parallel_rng_call(
    S0, K, T, r, sigma,
    n_simulations=1_000_000,
    n_workers=4,
    seed=42,
):
    """Price a European call using independent worker RNG streams."""
    if n_simulations < 2 or n_workers < 1:
        raise ValueError("Invalid simulation or worker count")
    if n_workers > n_simulations // 2:
        raise ValueError("Each worker requires at least two simulations")

    beta = optimal_beta(S0, K, T, r, sigma)

    seed_sequence = np.random.SeedSequence(seed)
    worker_seeds = seed_sequence.spawn(n_workers)

    base = n_simulations // n_workers
    remainder = n_simulations % n_workers

    tasks = []

    for i in range(n_workers):
        count = base + (1 if i < remainder else 0)

        tasks.append((
            S0, K, T, r, sigma,
            beta, count, worker_seeds[i]
        ))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_worker, tasks))

    total = sum(n * mean for n, mean, _ in results)
    mean = total / n_simulations

    # Pooled sample variance, including between-worker variation.
    total_ss = sum(
        (n - 1) * variance + n * (worker_mean - mean)**2
        for n, worker_mean, variance in results
    )

    sample_variance = total_ss / (n_simulations - 1)
    se = np.sqrt(sample_variance / n_simulations)

    return float(mean), float(se)
