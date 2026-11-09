
import time
import numpy as np

from numba import set_num_threads, get_num_threads

from optimised.european_control import optimal_beta
from optimised.parallel_control import parallel_control_call


PARAMS = (100, 100, 1, 0.05, 0.2)
N = 5_000_000
REPETITIONS = 7


def main():
    beta = optimal_beta(*PARAMS)
    maximum_threads = get_num_threads()

    counts = sorted(set(
        n for n in [1, 2, 4, 8, maximum_threads]
        if n <= maximum_threads
    ))

    # Compile before measuring.
    parallel_control_call(*PARAMS, beta, 1000)

    print(f"{'Threads':>8} {'Median ms':>12}")

    for threads in counts:
        set_num_threads(threads)
        times = []

        for _ in range(REPETITIONS):
            start = time.perf_counter()

            price, se = parallel_control_call(
                *PARAMS,
                beta,
                n_simulations=N,
                chunk_size=100_000_000,
                seed=42
            )

            times.append(
                1000 * (time.perf_counter() - start)
            )

        print(f"{threads:>8} {np.median(times):>12.3f}")


if __name__ == "__main__":
    main()
