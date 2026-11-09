
import numpy as np
from scipy.stats import norm


def monte_carlo_call(S0, K, T, r, sigma, n_simulations=100_000, seed=None):
    """
    Price a European call using Monte Carlo simulation.

    Returns:
        price: Estimated option price
        std_error: Standard error of the estimator
        ci: Approximate 95% confidence interval
    """
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if not isinstance(n_simulations, int) or n_simulations < 2:
        raise ValueError("n_simulations must be an integer >= 2")

    if T == 0:
        payoff = max(S0 - K, 0.0)
        return payoff, 0.0, (payoff, payoff)

    rng = np.random.default_rng(seed)

    # Simulate terminal asset prices under the risk-neutral measure
    Z = rng.standard_normal(n_simulations)

    ST = S0 * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z
    )

    # Discounted European call payoffs
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)

    price = np.mean(payoffs)

    # Sample standard deviation, with Bessel's correction
    std_error = np.std(payoffs, ddof=1) / np.sqrt(n_simulations)

    z = norm.ppf(0.975)
    ci = (price - z * std_error, price + z * std_error)

    return float(price), float(std_error), tuple(map(float, ci))


if __name__ == "__main__":
    price, se, ci = monte_carlo_call(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
        n_simulations=100_000,
        seed=42
    )

    print(f"Monte Carlo price: {price:.4f}")
    print(f"Standard error:    {se:.4f}")
    print(f"95% CI:            [{ci[0]:.4f}, {ci[1]:.4f}]")
