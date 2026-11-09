
from black_scholes import black_scholes_call
from monte_carlo import monte_carlo_call


S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

exact = black_scholes_call(S0, K, T, r, sigma)

print(f"Analytical price: {exact:.6f}\n")

for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    estimate, se, ci = monte_carlo_call(
        S0, K, T, r, sigma,
        n_simulations=n,
        seed=42
    )

    error = abs(estimate - exact)

    print(
        f"N={n:>8,} | "
        f"MC={estimate:.4f} | "
        f"Error={error:.4f} | "
        f"SE={se:.4f} | "
        f"CI covers exact={ci[0] <= exact <= ci[1]}"
    )
