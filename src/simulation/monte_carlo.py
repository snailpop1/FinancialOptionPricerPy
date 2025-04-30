# src/simulation/monte_carlo.py

import numpy as np
from typing import Tuple
import sys # Import sys for stderr warnings if needed

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    num_steps: int,
    num_simulations: int
) -> Tuple[np.ndarray, float]:
    """
    Simulates stock price paths using Geometric Brownian Motion (GBM).

    Args:
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying stock (annualized).
        T (float): Time to expiration (in years).
        num_steps (int): Number of time steps for simulation within each path.
        num_simulations (int): Number of simulation paths to generate.

    Returns:
        Tuple[np.ndarray, float]:
            - A numpy array of shape (num_simulations, num_steps + 1) containing
              the simulated stock price paths. The first column is S0.
            - The time step size dt.

    Raises:
        ValueError: If inputs are invalid (e.g., non-positive S0, sigma, T, steps, simulations).
    """
    # --- Input Validation ---
    if S0 <= 0:
        raise ValueError("Initial stock price (S0) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    # T=0 is handled in monte_carlo_european_call_price, but simulation requires T>0
    if T <= 0:
        raise ValueError("Time to expiration (T) must be positive for simulation.")
    if num_steps <= 0:
        raise ValueError("Number of steps must be positive.")
    if num_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")
    # r can theoretically be negative, but warn if extreme
    # if abs(r) > 0.5: print(f"Warning: Unusual risk-free rate r={r:.2f} provided.", file=sys.stderr)


    # --- Simulation Setup ---
    dt = T / num_steps  # Time step size

    # Create an array to store the paths.
    # Dimensions: simulations x (steps + 1 time points -> includes t=0)
    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = S0  # Set initial price S0 for all simulations at t=0

    # Generate random numbers from standard normal distribution N(0,1)
    # Dimensions: simulations x steps (one random shock per step per simulation)
    # Using numpy's default_rng for potentially better performance/randomness generation
    rng = np.random.default_rng()
    # Z represents the random shocks (Wiener process increments scaled by sqrt(dt))
    Z = rng.standard_normal((num_simulations, num_steps))

    # --- Precompute Constants for Efficiency ---
    # Drift term: (r - 0.5 * sigma^2) * dt
    drift = (r - 0.5 * sigma**2) * dt
    # Diffusion term: sigma * sqrt(dt)
    diffusion = sigma * np.sqrt(dt)

    # --- Simulate Paths Step-by-Step ---
    # Iterate through time steps from t=1 to num_steps
    for t in range(1, num_steps + 1):
        # Apply the discretized Geometric Brownian Motion formula:
        # S_t = S_{t-1} * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t )
        # Use vectorized operations for speed across all simulations simultaneously
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])
        # Note: Z[:, t-1] corresponds to the random shock for the interval ending at time t

    return paths, dt


def monte_carlo_european_call_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    num_steps: int,
    num_simulations: int
) -> float:
    """
    Estimates the price of a European call option using Monte Carlo simulation.

    Args:
        S0 (float): Initial stock price.
        K (float): Option strike price.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying stock (annualized).
        T (float): Time to expiration (in years).
        num_steps (int): Number of time steps for simulation within each path.
        num_simulations (int): Number of simulation paths to generate.

    Returns:
        float: The estimated price of the European call option.

    Raises:
        ValueError: If inputs passed to simulate_gbm_paths are invalid or K < 0 or T < 0.
    """
    # --- Input Validation (Basic checks) ---
    # More detailed validation (S0>0, sigma>0, steps>0, sims>0) is done in simulate_gbm_paths
    if K < 0:
        raise ValueError("Strike price (K) must be non-negative.")
    if T < 0:
         raise ValueError("Time to expiration (T) cannot be negative.")

    # --- Handle Edge Case: T=0 ---
    if T == 0:
        # At expiration, the price is simply the intrinsic value: max(S0 - K, 0)
        # No simulation needed.
        return max(S0 - K, 0.0)

    # --- Simulate Stock Price Paths ---
    try:
        # Call the GBM simulation function
        paths, _ = simulate_gbm_paths(S0, r, sigma, T, num_steps, num_simulations)
    except ValueError as e:
        # Propagate errors from the simulation function (e.g., invalid S0, sigma)
        raise e

    # --- Calculate Option Price from Paths ---
    # 1. Get the stock prices at expiration (T)
    #    This is the last column (index -1) of the paths array
    ST = paths[:, -1] # Terminal stock prices for all simulations

    # 2. Calculate the payoff for each path at expiration
    #    Payoff = max(S_T - K, 0) for a call option
    #    Using np.maximum is efficient for element-wise comparison on arrays
    payoffs = np.maximum(ST - K, 0)

    # 3. Calculate the average payoff across all simulated paths
    average_payoff = np.mean(payoffs)

    # 4. Discount the average payoff back to present value (t=0)
    #    Use the continuous risk-free rate discount factor: exp(-r * T)
    option_price = np.exp(-r * T) * average_payoff

    return option_price

# --- Example Usage ---
# This block runs only when the script is executed directly (e.g., python src/simulation/monte_carlo.py)
if __name__ == '__main__':
    # Define example parameters (consistent with other examples)
    S0_initial = 100.0       # Initial stock price
    K_strike_price = 105.0   # Strike price
    r_rate_annual = 0.05     # Risk-free rate (5%)
    sigma_volatility = 0.2   # Volatility (20%)
    T_maturity_years = 1.0   # Time to maturity (1 year)
    n_simulations = 10000    # Number of simulation paths
    m_steps = 100            # Number of time steps per path

    print(f"--- Monte Carlo Simulation Example ---")
    print(f"Input Parameters:")
    print(f"  Initial Stock Price (S0): {S0_initial:.2f}")
    print(f"  Strike Price (K):         {K_strike_price:.2f}")
    print(f"  Risk-Free Rate (r):       {r_rate_annual*100:.2f}%")
    print(f"  Volatility (sigma):       {sigma_volatility*100:.2f}%")
    print(f"  Time to Maturity (T):     {T_maturity_years:.2f} years")
    print(f"  Number of Simulations:    {n_simulations:,}") # Formatted
    print(f"  Number of Time Steps:     {m_steps:,}")      # Formatted

    try:
        # Calculate the price using the function
        mc_call_price = monte_carlo_european_call_price(
            S0_initial, K_strike_price, r_rate_annual, sigma_volatility,
            T_maturity_years, m_steps, n_simulations
        )
        print(f"\nEstimated European Call Price (Monte Carlo): {mc_call_price:.4f}") # Show more precision

        # --- Test Edge Case: T=0 ---
        print("\n--- Testing Edge Case T=0 ---")
        mc_price_T0_ITM = monte_carlo_european_call_price(
            S0=110, K=100, r=0.05, sigma=0.2, T=0, num_steps=10, num_simulations=100 # steps/sims irrelevant here
        )
        mc_price_T0_OTM = monte_carlo_european_call_price(
            S0=90, K=100, r=0.05, sigma=0.2, T=0, num_steps=10, num_simulations=100
        )
        print(f"Test Case T=0 (S=110, K=100): Price = {mc_price_T0_ITM:.4f} (Expected: 10.0)")
        print(f"Test Case T=0 (S=90, K=100): Price = {mc_price_T0_OTM:.4f} (Expected: 0.0)")

    except ValueError as e:
        # Catch and report any validation errors during calculation
        print(f"\nError during Monte Carlo simulation: {e}", file=sys.stderr)

