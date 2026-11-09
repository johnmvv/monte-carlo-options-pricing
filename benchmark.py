
import csv
import gc
import time
from pathlib import Path

import numpy as np

from black_scholes import black_scholes_call
from monte_carlo import monte_carlo_call
from antithetic import antithetic_call
from control_variate import control_variate_call


PARAMS = (100, 100, 1, 0.05, 0.2)

SAMPLE_SIZES = [
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
]

STATISTICAL_REPETITIONS = 500
TIMING_REPETITIONS = 15
TARGET_BATCH_SECONDS = 0.15
PILOT_FRACTION = 0.10

METHODS = ["Standard", "Antithetic", "Control variate"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_method(method, n, seed):
    """Return a single European call price estimate."""
    if method == "Standard":
        return monte_carlo_call(
            *PARAMS,
            n_simulations=n,
            seed=seed,
        )[0]

    if method == "Antithetic":
        return antithetic_call(
            *PARAMS,
            n_simulations=n,
            seed=seed,
        )[0]

    if method == "Control variate":
        pilot = max(2, round(n * PILOT_FRACTION))
        main = n - pilot

        return control_variate_call(
            *PARAMS,
            n_simulations=main,
            pilot_simulations=pilot,
            seed=seed,
        )[0]

    raise ValueError(f"Unknown method: {method}")


def statistical_benchmark(method, n, exact):
    """Estimate bias, variance, RMSE and their uncertainty."""
    estimates = np.empty(STATISTICAL_REPETITIONS)

    for i in range(STATISTICAL_REPETITIONS):
        estimates[i] = run_method(
            method,
            n,
            seed=100_000 + i,
        )

    errors = estimates - exact
    squared_errors = errors**2

    bias = float(np.mean(errors))
    variance = float(np.var(estimates, ddof=1))
    rmse = float(np.sqrt(np.mean(squared_errors)))

    # Monte Carlo uncertainty of the estimated bias
    bias_se = float(
        np.std(errors, ddof=1)
        / np.sqrt(STATISTICAL_REPETITIONS)
    )

    # Approximate uncertainty of RMSE using the delta method
    mse_se = float(
        np.std(squared_errors, ddof=1)
        / np.sqrt(STATISTICAL_REPETITIONS)
    )
    rmse_se = mse_se / (2 * rmse) if rmse > 0 else 0.0

    return {
        "bias": bias,
        "bias_se": bias_se,
        "variance": variance,
        "rmse": rmse,
        "rmse_se": rmse_se,
    }


def timing_benchmark(method, n):
    """Measure runtime per estimate using repeated batches."""

    # Warm up imports, allocation paths and numerical routines.
    for i in range(5):
        run_method(method, n, seed=i)

    # Calibrate batch size using a representative call.
    start = time.perf_counter()
    run_method(method, n, seed=999)
    single_time = time.perf_counter() - start

    batch_size = max(
        1,
        min(10_000, int(TARGET_BATCH_SECONDS / max(single_time, 1e-9))),
    )

    measurements = []

    for repetition in range(TIMING_REPETITIONS):
        # Keep garbage collection consistent during timing.
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()

        try:
            start = time.perf_counter()

            for i in range(batch_size):
                run_method(
                    method,
                    n,
                    seed=1_000_000 + repetition * batch_size + i,
                )

            elapsed = time.perf_counter() - start
        finally:
            if gc_enabled:
                gc.enable()

        measurements.append(
            elapsed / batch_size * 1000
        )

    measurements = np.asarray(measurements)

    return {
        "median_time_ms": float(np.median(measurements)),
        "q25_time_ms": float(np.percentile(measurements, 25)),
        "q75_time_ms": float(np.percentile(measurements, 75)),
        "batch_size": batch_size,
    }


def main():
    exact = black_scholes_call(*PARAMS)
    rows = []

    print(f"Analytical Black-Scholes price: {exact:.8f}\n")

    for n in SAMPLE_SIZES:
        print(f"\nSimulation budget: {n:,}")

        for method in METHODS:
            statistical = statistical_benchmark(method, n, exact)
            timing = timing_benchmark(method, n)

            row = {
                "method": method,
                "n_simulations": n,
                **statistical,
                **timing,
            }
            rows.append(row)

            print(
                f"{method:<17} "
                f"RMSE={statistical['rmse']:.6f} "
                f"Bias={statistical['bias']:+.6f} "
                f"Median={timing['median_time_ms']:.3f} ms"
            )

    output = RESULTS_DIR / "benchmark_results.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output}")

    # Estimate log-log convergence slopes.
    print("\nEstimated convergence slopes:")

    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]

        x = np.log([r["n_simulations"] for r in subset])
        y = np.log([r["rmse"] for r in subset])

        slope, intercept = np.polyfit(x, y, 1)

        print(f"{method:<17} slope = {slope:.4f}")

    return rows


if __name__ == "__main__":
    main()
