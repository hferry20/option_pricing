import numpy as np
from scipy.stats import norm
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from auxillary_functions import simulate_paths
from scipy.optimize import minimize

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
        self.option_type = option_type
        self.paths = None # Generated paths used to define the exercise-boundary
        self.pricingPaths = None # Generated paths used to price the option
        self.optimalBoundary = None
        self.cleanBoundary = None
        self.exerciseBoundary = None
        self.stoppingTimes = None
    
    # Payoff of the option at time t=tau, the stopping time given the stopping rules
    def payoff(self, S_tau):
        if self.option_type == "call":
            return np.maximum(S_tau - self.K, 0)
        elif self.option_type == "put":
            return np.maximum(self.K - S_tau, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
    def discounted_payoff(self, payoffs, stopping_times):
        return np.exp(-self.r * self.T * stopping_times/self.N) * payoffs
    
    def stopping_value(self):
        return self.pricingPaths[np.arange(self.sims), self.stoppingTimes]
    
    def exercise_boundary(self, points):
        t = np.linspace(0, self.T, self.N + 1)
        return np.interp(t, np.linspace(0, self.T, len(points)), points)
    
    def stopping_rules(self, paths, exercise_boundary):
        stopping_times = np.argmax(paths >= exercise_boundary, axis = 1)
        stopping_times[stopping_times == 0] = self.N  # If never reached K, stop at maturity
        return stopping_times

    def objective(self, points):
        B_t = self.exercise_boundary(points)
        stopping_times = self.stopping_rules(self.paths, B_t)
        S_tau = self.paths[np.arange(self.sims), stopping_times]
        payoffs = self.payoff(S_tau)
        discounted_payoff = self.discounted_payoff(payoffs, stopping_times)
        option_price = np.mean(discounted_payoff)

        # Constraints
        penalty_T = 1000 * (B_t[-1] - self.K)**2  # Penalty term for if B_T != K

        return -option_price + penalty_T

    def path_generation(self):
        paths = simulate_paths(self.S, self.mu, self.sigma, self.q, self.r, self.T, self.N, self.sims)
        return paths
    
    # Optimises boundary given n1 sample
    def optimize_boundary(self):
        if self.option_type == "call":
            initial_guess = np.linspace(self.S * (1 + 2*self.sigma), self.K, self.N+1)
        elif self.option_type == "put":
            initial_guess = np.linspace(self.S * (1 - 2*self.sigma), self.K, self.N+1)
        result = basinhopping(self.objective, x0=initial_guess, minimizer_kwargs={'method':'Powell'}, niter=int(round((N+1)/20,0)), disp = False) # REMEMBER TO CHANGE BACK TO /5
        return result.x
    
    # Cleans significant outliers as to not distort the fitting of the softer-trendline; avoids overfitting to n1 sample
    def cleaned_boundary(self, z_threshold):
        threshold = z_threshold * self.sigma * self.S
        cleaned_boundary = self.optimalBoundary.copy()
        for i in range(1, len(cleaned_boundary) - 10):
            if abs(cleaned_boundary[i] - cleaned_boundary[i-1]) > threshold:
                cleaned_boundary[i] = cleaned_boundary[i-1]
        cleaned_boundary[-1] = self.K
        if abs(cleaned_boundary[-2] - cleaned_boundary[-1]) > self.S * self.sigma * 1.5: # Checks to see if the boundary creation has gone nuts
            raise ValueError("Boundary creation corrupted")
        else:
            return cleaned_boundary
    
    # Sets the start-point at S_0 * (1 + 2 * vol) and end-point at K, minimisation function used in partnership with the line_fit() function
    def constrained_polynomial_fit(self, x, y, degree, start_value, end_value):
        def polynomial_obj(coeffs):
            poly = np.polyval(coeffs, x)
            return np.sum((poly - y) ** 2)

        def start_constraint(coeffs):
            return np.polyval(coeffs, x[0]) - start_value

        def end_constraint(coeffs):
            return np.polyval(coeffs, x[-1]) - end_value

        initial_guess = np.polyfit(x, y, degree)
        cons = [{'type': 'eq', 'fun': start_constraint}, {'type': 'eq', 'fun': end_constraint}]
        result = minimize(polynomial_obj, initial_guess, constraints=cons)
        return result.x

    # Fits the exercise-boundary in relation to the above function
    def line_fit(self):
        x = np.linspace(0, len(self.cleanBoundary) - 1, len(self.cleanBoundary))
        coeffs = self.constrained_polynomial_fit(x, self.cleanBoundary, 4, self.cleanBoundary[0], self.K)
        polynomial = np.polyval(coeffs, x)
        return polynomial

    def price(self):
        self.paths = self.path_generation()
        self.optimalBoundary = self.optimize_boundary()
        self.cleanBoundary = self.cleaned_boundary(2)
        self.exerciseBoundary = self.line_fit()
        self.pricingPaths = self.path_generation()
        self.stoppingTimes = self.stopping_rules(self.pricingPaths, self.exerciseBoundary)
        S_tau = self.stopping_value()
        payoffs = self.payoff(S_tau)
        discounted_payoffs = self.discounted_payoff(payoffs, self.stoppingTimes)
        price = np.mean(discounted_payoffs)
        return price
        
## TESTING

if __name__ == "__main__":
    S = 100     
    mu = 0.05   
    sigma = 0.2 
    q = 0  
    r = 0.03    
    K = 100     
    T = 1       
    N = 1000
    sims = 1000
    option_type = "call"

    american_option = AmericanOption(S, mu, sigma, q, r, K, T, N, sims, option_type)
    print(american_option.price())
    european_option = EuropeanOption(S,K,r,q,sigma,T,option_type)
    print(european_option.price())