
import time
import numpy as np

from optimised.european_control import optimised_control_call
from optimised.parallel_rng import parallel_rng_call


PARAMS = (100, 100, 1, 0.05, 0.2)
SIZES = [100_000, 1_000_000, 5_000_000, 10_000_000]
REPETITIONS = 9


def measure(func):
    func()  # Warm-up

    times = []

    for _ in range(REPETITIONS):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    return float(np.median(times))


def main():
    for n in SIZES:
        print(f"\nSimulation budget: {n:,}")

        serial_time = measure(
            lambda: optimised_control_call(
                *PARAMS, n_simulations=n, seed=42
            )
        )

        print(f"Serial: {serial_time:.3f} ms")

        for workers in [2, 4, 6, 12]:
            parallel_time = measure(
                lambda: parallel_rng_call(
                    *PARAMS,
                    n_simulations=n,
                    n_workers=workers,
                    seed=42
                )
            )

            print(
                f"{workers:>2} workers: "
                f"{parallel_time:.3f} ms "
                f"({serial_time / parallel_time:.2f}x)"
            )


if __name__ == "__main__":
    main()
