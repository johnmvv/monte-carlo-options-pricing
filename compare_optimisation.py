
import time
import numpy as np

from control_variate import control_variate_call
from black_scholes import black_scholes_call
from optimised.european_control import optimised_control_call


PARAMS = (100, 100, 1, 0.05, 0.2)
BUDGETS = [1_000, 10_000, 100_000, 1_000_000]
REPETITIONS = 100


def baseline(n, seed):
    pilot = max(2, n // 10)
    return control_variate_call(
        *PARAMS,
        n_simulations=n - pilot,
        pilot_simulations=pilot,
        seed=seed
    )


def optimised(n, seed):
    return optimised_control_call(
        *PARAMS,
        n_simulations=n,
        seed=seed
    )


def evaluate(method, n, exact):
    estimates = []
    times = []

    for seed in range(REPETITIONS):
        start = time.perf_counter()
        result = method(n, seed)
        times.append(time.perf_counter() - start)
        estimates.append(result[0])

    estimates = np.asarray(estimates)

    return (
        np.sqrt(np.mean((estimates - exact)**2)),
        np.median(times) * 1000
    )


def main():
    exact = black_scholes_call(*PARAMS)

    print(
        f"{'Paths':>10} {'Old RMSE':>12} {'New RMSE':>12} "
        f"{'Old ms':>10} {'New ms':>10} {'Speedup':>9}"
    )

    # Compile and warm up the optimised implementation
    optimised(1_000, seed=42)
    baseline(1_000, seed=42)

    for n in BUDGETS:
        old_rmse, old_time = evaluate(baseline, n, exact)
        new_rmse, new_time = evaluate(optimised, n, exact)

        print(
            f"{n:>10,} {old_rmse:>12.6f} "
            f"{new_rmse:>12.6f} "
            f"{old_time:>10.3f} {new_time:>10.3f} "
            f"{old_time/new_time:>8.2f}x"
        )


if __name__ == "__main__":
    main()
