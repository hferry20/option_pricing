import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from auxillary_functions import simulate_paths, stopping_rule, payoff, objective

def european_option(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str) -> float:
    d1 = (1/(sigma * np.sqrt(T))) * (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        pi = np.exp(-r*T) * (S * np.exp((r-q)*T) * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        pi = np.exp(-r*T) * (K * norm.cdf(-d2) - S * np.exp((r-q)*T) * norm.cdf(-d1))
    return pi

def american_option(S: float, K: float, mu: float, sigma: float, q: float, r: float, T: float, N: int, sims: int, option_type: str) -> float:
    # Garcia (2001) - Generation of exercise boundary via parameterisation and optimisation
    paths, discounted_paths, time_grid = simulate_paths(S, mu, sigma, q, r, T, N, sims)

    