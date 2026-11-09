
from asian_options import asian_call


PARAMS = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "sigma": 0.2,
    "n_simulations": 100_000,
    "n_steps": 12,
    "seed": 42,
}

standard = asian_call(**PARAMS)
antithetic = asian_call(**PARAMS, antithetic=True)

print("Arithmetic Asian call (monthly monitoring)\n")

print(f"{'Method':<18} {'Price':>12} {'Std Error':>12}")
print("-" * 44)

for name, result in [
    ("Standard MC", standard),
    ("Antithetic", antithetic)
]:
    print(f"{name:<18} {result[0]:>12.6f} {result[1]:>12.6f}")

print(
    f"\nEstimated variance reduction factor: "
    f"{(standard[1] / antithetic[1])**2:.2f}x"
)
