import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from auxillary_functions import simulate_paths
from random import randint

class EuropeanOption:
    def __init__(self, S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.option_type = option_type

    def d1(self):
        return (1/(self.sigma * np.sqrt(self.T))) * (np.log(self.S/self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T)

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def price(self):
        d1 = self.d1()
        d2 = self.d2()
        if self.option_type == "call":
            price = np.exp(-self.r * self.T) * (self.S * np.exp((self.r - self.q) * self.T) * norm.cdf(d1) - self.K * norm.cdf(d2))
        elif self.option_type == "put":
            price = np.exp(-self.r * self.T) * (self.K * norm.cdf(-d2) - self.S * np.exp((self.r - self.q) * self.T) * norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        return price

class AmericanOption:
    def __init__(self, S: float, mu: float, sigma: float, q: float, r: float, K: float, T: float, N: int, sims: int, option_type: str):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.q = q
        self.r = r
        self.K = K
        self.T = T
        self.N = N  # Number of steps from t=0 to t=T
        self.sims = sims
        self.paths = None
        self.option_type = option_type
    
    # Payoff of the option at time t=tau, the stopping time given the stopping rules
    def payoff(self, S_tau):
        if self.option_type == "Call":
            return np.maximum(S_tau - self.K, 0)
        elif self.option_type == "Put":
            return np.maximum(self.K - S_tau, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
    def discounted_payoff(self, payoffs, stopping_times):
        return np.exp(-self.r * self.T * stopping_times/self.N) * payoffs
    
    def exercise_boundary(self, points):
        t = np.linspace(0, self.T, self.N + 1)
        return np.interp(t, np.linspace(0, self.T, len(points)), points)
    
    def stopping_rules(self, exercise_boundary):
        stopping_times = np.argmax(self.paths >= exercise_boundary, axis = 1)
        stopping_times[stopping_times == 0] = self.N  # If never reached K, stop at maturity
        return stopping_times

    def objective(self, points):
        # Finding option price given some exercise boundary
        B_t = self.exercise_boundary(points)
        stopping_times = self.stopping_rules(B_t)
        S_tau = self.paths[np.arrange(self.sims), stopping_times]
        payoffs = self.payoff(S_tau)
        discounted_payoff = discounted_payoff(payoffs, stopping_times)
        option_price = np.mean(discounted_payoff)

        # Constraints
        penalty_T = 1000 * (B_t[-1] - self.K)**2  # Penalty term for if B_T = K

        return -option_price + penalty_T
    
    def optimize_coeffs(self, intial_guess):
        result = minimize(self.objective, x0 = intial_guess)
        return result.x
    

## TESTING

if __name__ == "__main__":
    S = 100     
    mu = 0.05   
    sigma = 0.2 
    q = 0  
    r = 0.03    
    K = 100     
    T = 1       
    N = 100     
    sims = 10 
    option_type = "Call"

    paths = simulate_paths(S, mu, sigma, q, r, T, N, sims)

    american_option = AmericanOption(S, mu, sigma, q, r, K, T, N, sims, option_type)
    american_option.paths = simulate_paths(S, mu, sigma, q, r, T, N, sims)
    points = np.linspace(150,100,N+1)
    exercise_boundary = american_option.exercise_boundary(points)

    paths = american_option.paths
    stopping_point = american_option.stopping_rules(exercise_boundary)
    z = np.linspace(0, T, N+1)
    for i in range(sims):
        print("Sim:", i)
        x = np.argmax(paths[i] >= exercise_boundary, axis = 0)
        print(x)
        print(paths[i, x-1], paths[i,x])
        print(exercise_boundary[x-1], exercise_boundary[x])
        
        print(stopping_point[i])
        y = paths[i]
        plt.plot(z,y, label = 'Asset {i}'.format(i=i))
    plt.plot(z, exercise_boundary)
    plt.legend()
    plt.show()

    

