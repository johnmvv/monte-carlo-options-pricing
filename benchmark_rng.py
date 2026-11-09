
import gc
import time

import numpy as np

from optimised.european_control import (
    optimal_beta,
    _pricing_kernel,
    _pricing_kernel_from_normals,
)


PARAMS = (100, 100, 1, 0.05, 0.2)
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
REPETITIONS = 15


def measure(func, repetitions=REPETITIONS):
    """Return median runtime in milliseconds."""
    times = []

    for _ in range(repetitions):
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()

        try:
            start = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - start
        finally:
            if gc_enabled:
                gc.enable()

        times.append(elapsed * 1000)

    return float(np.median(times))


def main():
    S0, K, T, r, sigma = PARAMS
    beta = optimal_beta(*PARAMS)

    print(
        f"{'Paths':>10} "
        f"{'Numba RNG':>12} "
        f"{'NumPy + Numba':>15} "
        f"{'Speedup':>10} "
        f"{'Kernel only':>13}"
    )
    print("-" * 70)

    for n in SIZES:
        # Pre-generated input for the kernel-only benchmark
        rng = np.random.default_rng(42)
        normals = rng.standard_normal(n)

        # Compile both kernels before timing
        _pricing_kernel(
            S0, K, T, r, sigma, beta, n, 42
        )
        _pricing_kernel_from_normals(
            S0, K, T, r, sigma, beta, normals
        )

        # Method A: Numba generates normals internally
        def numba_rng():
            return _pricing_kernel(
                S0, K, T, r, sigma, beta, n, 42
            )

        # Method B: NumPy generates normals, Numba prices
        def numpy_numba():
            local_rng = np.random.default_rng(42)
            z = local_rng.standard_normal(n)

            return _pricing_kernel_from_normals(
                S0, K, T, r, sigma, beta, z
            )

        # Method C: Numerical kernel only, no RNG
        def kernel_only():
            return _pricing_kernel_from_normals(
                S0, K, T, r, sigma, beta, normals
            )

        time_a = measure(numba_rng)
        time_b = measure(numpy_numba)
        time_c = measure(kernel_only)

        print(
            f"{n:>10,} "
            f"{time_a:>12.3f} "
            f"{time_b:>15.3f} "
            f"{time_a/time_b:>9.2f}x "
            f"{time_c:>13.3f}"
        )


if __name__ == "__main__":
    main()
