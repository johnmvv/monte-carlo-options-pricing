
import numpy as np

from black_scholes import black_scholes_call
from monte_carlo import monte_carlo_call
from antithetic import antithetic_call


S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N = 100_000
REPETITIONS = 100

exact = black_scholes_call(S0, K, T, r, sigma)

standard_estimates = []
antithetic_estimates = []

for seed in range(REPETITIONS):
    standard, _, _ = monte_carlo_call(
        S0, K, T, r, sigma,
        n_simulations=N,
        seed=seed
    )

    antithetic, _, _ = antithetic_call(
        S0, K, T, r, sigma,
        n_simulations=N,
        seed=seed
    )

    standard_estimates.append(standard)
    antithetic_estimates.append(antithetic)

standard_variance = np.var(standard_estimates, ddof=1)
antithetic_variance = np.var(antithetic_estimates, ddof=1)

print(f"Analytical price: {exact:.6f}")
print(f"Standard MC variance: {standard_variance:.8f}")
print(f"Antithetic variance:  {antithetic_variance:.8f}")

print(
    f"Variance reduction factor: "
    f"{standard_variance / antithetic_variance:.2f}x"
)

print(
    f"Standard MC RMSE: "
    f"{np.sqrt(np.mean((np.array(standard_estimates) - exact)**2)):.6f}"
)

print(
    f"Antithetic RMSE: "
    f"{np.sqrt(np.mean((np.array(antithetic_estimates) - exact)**2)):.6f}"
)
