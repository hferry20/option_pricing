import numpy as np

# GBM path-simulation
def simulate_paths(S: float, mu: float, sigma: float, q: float, r: float, T: float, N: int, sims: int):
    paths = np.zeros((sims, N+1))
    paths[:,0] = S
    dt = float(T/N)
    for t in range(1, N+1):
        paths[:, t] = paths[:, t-1] * np.exp((mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(sims))
    return paths

# Density function of a normal distribution with (mu, sigma^2)
def norm_density(x, mu, sigma):
    return 1/(sigma * np.sqrt(np.pi * 2)) * np.exp(-((x-mu)**2)/(2*sigma**2))

# Density function of an absorbed Weiner process
# Alpha = X(0), Beta = Absoption level
def absorbed_density(x, mu, sigma, t, alpha, beta):
    return norm_density(x, mu*t + alpha, sigma*np.sqrt(t)) - np.exp(-(2*mu*(alpha-beta))/sigma**2) * norm_density(x, mu*t - alpha + 2*beta, sigma*np.sqrt(t))