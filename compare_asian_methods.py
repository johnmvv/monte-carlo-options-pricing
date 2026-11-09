
from asian_options import asian_call
from asian_control_variate import asian_control_variate_call
from geometric_asian import geometric_asian_call


PARAMS = dict(
    S0=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    n_steps=12,
)

TOTAL_PATHS = 100_000
PILOT_PATHS = 10_000
SEED = 42

standard = asian_call(
    **PARAMS,
    n_simulations=TOTAL_PATHS,
    seed=SEED
)

antithetic = asian_call(
    **PARAMS,
    n_simulations=TOTAL_PATHS,
    antithetic=True,
    seed=SEED
)

control = asian_control_variate_call(
    **PARAMS,
    n_simulations=TOTAL_PATHS - PILOT_PATHS,
    pilot_simulations=PILOT_PATHS,
    seed=SEED
)

print("Arithmetic Asian call pricing\n")
print(f"Geometric analytical benchmark: "
      f"{geometric_asian_call(**PARAMS):.6f}\n")

print(f"{'Method':<20} {'Price':>12} {'Std Error':>12}")
print("-" * 46)

for name, result in [
    ("Standard MC", standard),
    ("Antithetic", antithetic),
    ("Geometric CV", control)
]:
    print(
        f"{name:<20} "
        f"{result[0]:>12.6f} "
        f"{result[1]:>12.6f}"
    )

print(f"\nEstimated control coefficient: {control[3]:.6f}")
print(
    "Geometric CV vs standard variance ratio: "
    f"{(standard[1] / control[1])**2:.2f}x"
)
