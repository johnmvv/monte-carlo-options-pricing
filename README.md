
# Monte Carlo Options Pricing Engine

A Python implementation of numerical option pricing under the Black–Scholes model, with a focus on Monte Carlo simulation, variance reduction and computational performance.

## Overview

This project explores the pricing of financial derivatives using stochastic simulation and numerical methods.

The objective is to implement option-pricing algorithms from first principles, validate them against analytical results, and investigate the trade-off between statistical accuracy and computational cost.

**Status:** In development.

## Mathematical Framework

Under the risk-neutral Black–Scholes model, the underlying asset follows geometric Brownian motion:

$$
dS_t = rS_t\,dt + \sigma S_t\,dW_t,
$$

where:

- $S_t$ is the asset price at time $t$.
- $r$ is the continuously compounded risk-free interest rate.
- $\sigma$ is the volatility.
- $W_t$ is standard Brownian motion.

Assuming constant parameters and no dividends, the terminal asset price is

$$
S_T = S_0 \exp\left((r-\tfrac12\sigma^2)T
+ \sigma\sqrt{T}Z\right),
\qquad Z\sim N(0,1).
$$

For a European call option with strike $K$ and maturity $T$:

$$
C_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[(S_T-K)^+].
$$

The Monte Carlo estimator is

$$
\widehat{C}_N =
\frac{e^{-rT}}{N}
\sum_{i=1}^{N}(S_T^{(i)}-K)^+.
$$

The project investigates the estimator's convergence, sampling variance and computational efficiency.

## Features

### Initial implementation

- [ ] Analytical Black–Scholes pricing for European options
- [ ] Monte Carlo pricing using geometric Brownian motion
- [ ] Confidence intervals and standard errors
- [ ] Automated numerical validation

### Planned extensions

- [ ] Antithetic variates
- [ ] Control variates
- [ ] Arithmetic Asian option pricing
- [ ] Convergence analysis and visualisation
- [ ] Runtime benchmarking and Numba optimisation

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- Numba
- pytest

## Validation

Numerical results will be compared against analytical Black–Scholes prices where closed-form solutions are available.

Planned tests include:

1. Agreement with analytical European option prices.
2. Monte Carlo convergence as sample size increases.
3. Confidence interval coverage across repeated simulations.
4. Correct behaviour at maturity and zero volatility.
5. Variance reduction relative to standard Monte Carlo.

## Example Parameters

| Parameter | Value |
|---|---:|
| Initial asset price | 100 |
| Strike price | 100 |
| Time to maturity | 1 year |
| Risk-free rate | 5% |
| Volatility | 20% |

For these parameters, the analytical European call price is approximately **10.4506**.

This provides a benchmark for the numerical implementations.

## Project Structure

The repository structure will be updated as components are implemented.

## Limitations

The model assumes geometric Brownian motion, constant volatility, constant interest rates and frictionless markets.

The implementation is intended for mathematical and computational study, not production trading.

## Licence

Released under the MIT Licence.
