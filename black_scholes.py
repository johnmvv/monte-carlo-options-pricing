
import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, T, r, sigma):
    """Return the Black-Scholes price of a European call.

    Assumes no dividends and constant risk-free rate and volatility.
    S0 > 0, K > 0, T >= 0, sigma >= 0.
    """
    if S0 <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Invalid option parameters")

    if T == 0:
        return max(S0 - K, 0.0)

    if sigma == 0:
        return np.exp(-r * T) * max(S0 * np.exp(r * T) - K, 0.0)

    d1 = (
        np.log(S0 / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


if __name__ == "__main__":
    price = black_scholes_call(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2
    )
    print(f"European call price: {price:.4f}")
