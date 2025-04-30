# src/models/black_scholes.py

import numpy as np
from scipy.stats import norm
from typing import Tuple
import sys # Import sys for stderr warnings if needed

def calculate_d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    """
    Calculates the d1 and d2 parameters used in the Black-Scholes formula.

    Args:
        S (float): Current stock price.
        K (float): Option strike price.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying stock (annualized).
        T (float): Time to expiration (in years).

    Returns:
        Tuple[float, float]: A tuple containing d1 and d2.

    Raises:
        ValueError: If sigma or T is non-positive, or if S or K is negative.
    """
    # Validate inputs
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    # T=0 is handled in the european_call_price function, but d1/d2 calculation requires T>0
    if T <= 0:
         raise ValueError("Time to expiration (T) must be positive for d1/d2 calculation.")
    if S < 0 or K < 0:
        raise ValueError("Stock price (S) and strike price (K) must be non-negative.")

    # Check for edge case S=0
    if S == 0:
        # If stock price is zero, d1 goes to negative infinity.
        # N(d1) becomes 0. Call price is 0.
        # Handle this in the main price function. Return -inf to signal this.
        return -np.inf, -np.inf

    # Calculate intermediate term: sigma * sqrt(T)
    sigma_sqrt_T = sigma * np.sqrt(T)

    # Prevent division by zero if K is zero (unlikely for standard options but handled)
    if K == 0:
         # If strike is 0, call option is worth S (guaranteed exercise)
         # d1/d2 become infinite. Handle in main function. Return inf.
         return np.inf, np.inf

    # Calculate d1 and d2 using the standard formula
    # Add a small epsilon to prevent log(0) if S/K is extremely small, although S=0 is handled
    # numerator = np.log(S / K + 1e-10) + (r + 0.5 * sigma**2) * T
    numerator = np.log(S / K) + (r + 0.5 * sigma**2) * T
    denominator = sigma_sqrt_T

    # Check for potential division by zero if sigma_sqrt_T is somehow zero (shouldn't happen with T>0, sigma>0)
    if denominator == 0:
        # This case indicates an issue with inputs not caught earlier or numerical instability
        raise ValueError("Denominator (sigma * sqrt(T)) is zero, cannot calculate d1/d2.")

    d1 = numerator / denominator
    d2 = d1 - denominator # d2 = d1 - sigma*sqrt(T)
    return d1, d2

def european_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculates the price of a European call option using the Black-Scholes formula.

    Args:
        S (float): Current stock price.
        K (float): Option strike price.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying stock (annualized).
        T (float): Time to expiration (in years).

    Returns:
        float: The estimated price of the European call option.

    Raises:
        ValueError: If inputs are invalid (e.g., negative prices, non-positive sigma/T).
    """
    # --- Input Validation ---
    if S < 0 or K < 0:
        raise ValueError("Stock price (S) and strike price (K) must be non-negative.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    if T < 0:
         raise ValueError("Time to expiration (T) cannot be negative.")
    # Optional: Add a check/warning for very high volatility or unusual rates
    # if sigma > 2.0: print(f"Warning: High volatility sigma={sigma:.2f} provided.", file=sys.stderr)
    # if abs(r) > 0.5: print(f"Warning: Unusual risk-free rate r={r:.2f} provided.", file=sys.stderr)


    # --- Handle Edge Cases ---
    # Case 1: Time to expiration is zero
    if T == 0:
        # At expiration, the price is simply the intrinsic value: max(S - K, 0)
        return max(S - K, 0.0)
    # Case 2: Stock price is zero
    if S == 0:
        # If the stock price is zero, the call option is worthless
        return 0.0
    # Case 3: Strike price is zero
    if K == 0:
        # If strike is zero, payoff is S_T. Assuming no dividends, the present
        # value under risk-neutral measure is S_0 * exp(rT) * exp(-rT) = S_0.
        # (If dividends q, it's S_0 * exp(-qT))
        # For simplicity without dividends:
        return S # Price is just the current stock price

    # --- Standard Calculation (T > 0, S > 0, K > 0) ---
    try:
        # Calculate d1 and d2 parameters
        d1, d2 = calculate_d1_d2(S, K, r, sigma, T)

        # Check if d1/d2 calculation resulted in non-finite values (e.g., inf due to K=0 handled above, but check anyway)
        if not (np.isfinite(d1) and np.isfinite(d2)):
             # This case should ideally not be reached due to prior checks
             raise ValueError(f"d1 ({d1}) or d2 ({d2}) calculation resulted in non-finite values. Check inputs.")

    except ValueError as e:
        # Propagate errors from d1/d2 calculation (e.g., T<=0 if not caught earlier)
        raise e # Re-raise the specific error

    # Calculate the call price using the Black-Scholes formula components:
    # N(d1) and N(d2) are CDF of standard normal distribution
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    # Discount factor for strike price
    discount_factor = np.exp(-r * T)

    # The formula: C = S * N(d1) - K * exp(-rT) * N(d2)
    price = (S * N_d1) - (K * discount_factor * N_d2)

    # Ensure price is not negative due to potential floating point inaccuracies
    # Black-Scholes price should theoretically always be non-negative.
    return max(price, 0.0)

# --- Example Usage ---
# This block runs only when the script is executed directly (e.g., python src/models/black_scholes.py)
if __name__ == '__main__':
    # Define example parameters
    S0 = 100.0       # Initial stock price
    K_strike = 105.0 # Strike price
    r_rate = 0.05    # Risk-free rate (5%)
    sigma_vol = 0.2  # Volatility (20%)
    T_maturity = 1.0 # Time to maturity (1 year)

    print(f"--- Black-Scholes Calculation Example ---")
    print(f"Input Parameters:")
    print(f"  Stock Price (S):    {S0:.2f}")
    print(f"  Strike Price (K):   {K_strike:.2f}")
    print(f"  Risk-Free Rate (r): {r_rate*100:.2f}%")
    print(f"  Volatility (sigma): {sigma_vol*100:.2f}%")
    print(f"  Time to Maturity (T): {T_maturity:.2f} years")

    try:
        # Calculate the price using the function
        call_price_bs = european_call_price(S0, K_strike, r_rate, sigma_vol, T_maturity)
        print(f"\nCalculated European Call Price: {call_price_bs:.4f}") # Show more precision

        # --- Test Edge Cases ---
        print("\n--- Testing Edge Cases ---")
        # Test T=0 (At-the-money, In-the-money, Out-of-the-money)
        price_T0_ATM = european_call_price(S=100, K=100, r=0.05, sigma=0.2, T=0)
        price_T0_ITM = european_call_price(S=110, K=100, r=0.05, sigma=0.2, T=0)
        price_T0_OTM = european_call_price(S=90, K=100, r=0.05, sigma=0.2, T=0)
        print(f"Test Case T=0 (S=100, K=100): Price = {price_T0_ATM:.4f} (Expected: 0.0)")
        print(f"Test Case T=0 (S=110, K=100): Price = {price_T0_ITM:.4f} (Expected: 10.0)")
        print(f"Test Case T=0 (S=90, K=100): Price = {price_T0_OTM:.4f} (Expected: 0.0)")

        # Test S=0
        price_S0 = european_call_price(S=0, K=100, r=0.05, sigma=0.2, T=1)
        print(f"Test Case S=0 (K=100, T=1): Price = {price_S0:.4f} (Expected: 0.0)")

        # Test K=0
        price_K0 = european_call_price(S=100, K=0, r=0.05, sigma=0.2, T=1)
        print(f"Test Case K=0 (S=100, T=1): Price = {price_K0:.4f} (Expected: {S0:.1f})")

    except ValueError as e:
        # Catch and report any validation errors during calculation
        print(f"\nError calculating Black-Scholes price: {e}", file=sys.stderr)

