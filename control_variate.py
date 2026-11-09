
import numpy as np
from scipy.stats import norm


def control_variate_call(
    S0, K, T, r, sigma,
    n_simulations=100_000,
    pilot_simulations=10_000,
    seed=None
):
    """
    Price a European call using a control variate.

    Uses discounted terminal stock value as the control.
    Estimates beta using an independent pilot sample.

    Returns:
        price, standard_error, confidence_interval, beta
    """
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    for n in (n_simulations, pilot_simulations):
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("Simulation counts must be integers >= 2")

    if T == 0 or sigma == 0:
        ST = S0 * np.exp(r * T)
        price = np.exp(-r * T) * max(ST - K, 0.0)
        return float(price), 0.0, (float(price), float(price)), 0.0

    rng = np.random.default_rng(seed)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    def sample(n):
        Z = rng.standard_normal(n)
        ST = S0 * np.exp(drift + diffusion * Z)

        X = discount * np.maximum(ST - K, 0.0)
        Y = discount * ST

        return X, Y

    # Independent pilot sample for beta estimation
    X_pilot, Y_pilot = sample(pilot_simulations)

    covariance = np.cov(X_pilot, Y_pilot, ddof=1)[0, 1]
    variance_y = np.var(Y_pilot, ddof=1)

    beta = covariance / variance_y

    # Main pricing sample
    X, Y = sample(n_simulations)

    adjusted_payoffs = X - beta * (Y - S0)

    price = np.mean(adjusted_payoffs)
    std_error = np.std(adjusted_payoffs, ddof=1) / np.sqrt(n_simulations)

    z = norm.ppf(0.975)
    ci = (
        price - z * std_error,
        price + z * std_error
    )

    return (
        float(price),
        float(std_error),
        tuple(map(float, ci)),
        float(beta)
    )
