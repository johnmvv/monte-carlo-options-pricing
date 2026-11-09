
import gc
import time

import numpy as np

from numba_asian import (
    asian_payoffs_numba,
    asian_payoffs_numpy,
)


PARAMS = (100, 100, 1, 0.05, 0.2)
PATH_COUNTS = [1_000, 10_000, 100_000, 500_000]
N_STEPS = 12
REPETITIONS = 15


def measure(func, normals):
    """Return median execution time in milliseconds."""
    # Warm-up also excludes Numba compilation time.
    func(*PARAMS, normals)

    times = []

    for _ in range(REPETITIONS):
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()

        try:
            start = time.perf_counter()
            result = func(*PARAMS, normals)
            elapsed = time.perf_counter() - start
        finally:
            if gc_enabled:
                gc.enable()

        # Ensure the result remains observable.
        assert np.isfinite(result).all()

        times.append(elapsed * 1000)

    return float(np.median(times))


def main():
    rng = np.random.default_rng(42)

    print(
        f"{'Paths':>10} "
        f"{'NumPy (ms)':>14} "
        f"{'Numba (ms)':>14} "
        f"{'Speedup':>10}"
    )
    print("-" * 52)

    for n in PATH_COUNTS:
        normals = rng.standard_normal((n, N_STEPS))

        numpy_time = measure(asian_payoffs_numpy, normals)
        numba_time = measure(asian_payoffs_numba, normals)

        print(
            f"{n:>10,} "
            f"{numpy_time:>14.3f} "
            f"{numba_time:>14.3f} "
            f"{numpy_time / numba_time:>9.2f}x"
        )


if __name__ == "__main__":
    main()
