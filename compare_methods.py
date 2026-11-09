
from black_scholes import black_scholes_call
from monte_carlo import monte_carlo_call
from antithetic import antithetic_call
from control_variate import control_variate_call


S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N = 100_000
SEED = 42

exact = black_scholes_call(S0, K, T, r, sigma)

standard = monte_carlo_call(
    S0, K, T, r, sigma, n_simulations=N, seed=SEED
)

antithetic = antithetic_call(
    S0, K, T, r, sigma, n_simulations=N, seed=SEED
)

control = control_variate_call(
    S0, K, T, r, sigma,
    n_simulations=N - 10_000,
    pilot_simulations=10_000,
    seed=SEED
)

print(f"Analytical Black-Scholes price: {exact:.6f}\n")

print(f"{'Method':<20} {'Estimate':>12} {'Std Error':>12} {'Abs Error':>12}")
print("-" * 60)

methods = [
    ("Standard MC", standard),
    ("Antithetic", antithetic),
    ("Control Variate", control)
]

for name, result in methods:
    price, se = result[:2]
    print(
        f"{name:<20} "
        f"{price:>12.6f} "
        f"{se:>12.6f} "
        f"{abs(price - exact):>12.6f}"
    )

print(f"\nEstimated control coefficient beta: {control[3]:.6f}")
