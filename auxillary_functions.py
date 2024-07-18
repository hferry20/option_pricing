import numpy as np

# GBM path-simulation
def simulate_paths(S: float, mu: float, sigma: float, q: float, r: float, T: float, N: int, sims: int):
    paths = np.zeros((sims, N+1))
    discounted_paths = np.zeros((sims, N+1))
    paths[:,0] = S
    discounted_paths[:, 0] = S
    dt = float(T/N)
    time_grid = np.linspace(0, T, N+1)
    for t in range(1, N+1):
        paths[:, t] = paths[:, t-1] * np.exp((mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(sims))
        discounted_paths[:, t] = paths[:, t] * np.exp(-r*t/N*T)
    return paths, discounted_paths, time_grid

# Functions needed for American option exercise boundary parameterisation
