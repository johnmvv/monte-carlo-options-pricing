
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
df = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")

methods = df["method"].unique()

# Plot 1: RMSE convergence
fig, ax = plt.subplots(figsize=(8, 5))

for method in methods:
    data = df[df["method"] == method]
    ax.loglog(
        data["n_simulations"],
        data["rmse"],
        marker="o",
        label=method
    )

ax.set_xlabel("Total simulated terminal prices")
ax.set_ylabel("Empirical RMSE")
ax.set_title("Monte Carlo Pricing: Convergence")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "convergence.png", dpi=200)
plt.close(fig)


# Plot 2: Variance reduction
pivot = df.pivot(
    index="n_simulations",
    columns="method",
    values="variance"
)

fig, ax = plt.subplots(figsize=(8, 5))

for method in ["Antithetic", "Control variate"]:
    reduction = pivot["Standard"] / pivot[method]
    ax.semilogx(
        reduction.index,
        reduction.values,
        marker="o",
        label=method
    )

ax.axhline(1, linestyle="--", label="No reduction")
ax.set_xlabel("Total simulated terminal prices")
ax.set_ylabel("Variance reduction factor")
ax.set_title("Variance Reduction Relative to Standard MC")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "variance_reduction.png", dpi=200)
plt.close(fig)


# Plot 3: Runtime versus accuracy
fig, ax = plt.subplots(figsize=(8, 5))

for method in methods:
    data = df[df["method"] == method]
    ax.loglog(
        data["median_time_ms"],
        data["rmse"],
        marker="o",
        label=method
    )

ax.set_xlabel("Median runtime per estimate (ms)")
ax.set_ylabel("Empirical RMSE")
ax.set_title("Computational Efficiency of Pricing Methods")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "runtime_accuracy.png", dpi=200)
plt.close(fig)

print("Saved three plots to results/")
