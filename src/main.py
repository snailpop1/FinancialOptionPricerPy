# src/main.py

import argparse
import sys
import time # To measure execution time
# --- Import model and simulation functions ---
# Use absolute imports relative to the 'src' directory when running as a module
# Example: python -m src.main ...
try:
    # Assumes these files exist in the correct subdirectories relative to src/
    from models.black_scholes import european_call_price as black_scholes_call
    from simulation.monte_carlo import monte_carlo_european_call_price as monte_carlo_call
    # --- Import the volatility predictor ---
    from volatility.predictor import VolatilityPredictor
except ImportError as e:
    # Provide a helpful error message if imports fail
    print(f"Error importing required modules: {e}", file=sys.stderr)
    print("Please ensure the script is run from the project root directory (the one containing the 'src' folder)", file=sys.stderr)
    print("using the command: 'python -m src.main [arguments]'", file=sys.stderr)
    print("Also verify that all necessary files (black_scholes.py, monte_carlo.py, predictor.py)", file=sys.stderr)
    print("and the empty __init__.py files exist in the correct subdirectories (src/, src/models/, src/simulation/, src/volatility/).", file=sys.stderr)
    sys.exit(1) # Exit if imports fail


def parse_arguments():
    """Parses command-line arguments for the option pricer."""
    parser = argparse.ArgumentParser(
        description="European Call Option Pricer using Black-Scholes, Monte Carlo, and optional ML Volatility Prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help message
    )

    # --- Option Parameters ---
    option_group = parser.add_argument_group('Option Parameters')
    option_group.add_argument(
        "-s", "--stock", type=float, required=True, metavar='PRICE',
        help="Current underlying stock price."
    )
    option_group.add_argument(
        "-k", "--strike", type=float, required=True, metavar='PRICE',
        help="Option strike price."
    )
    option_group.add_argument(
        "-r", "--rate", type=float, required=True, metavar='RATE',
        help="Risk-free interest rate (annualized decimal, e.g., 0.05 for 5%)."
    )
    option_group.add_argument(
        "-t", "--time", type=float, required=True, metavar='YEARS',
        help="Time to expiration in years (e.g., 0.5 for 6 months)."
    )

    # --- Volatility Source ---
    vol_group = parser.add_argument_group('Volatility Specification (Choose One)')
    # Use a mutually exclusive group to ensure only one volatility source is chosen
    vol_mutex_group = vol_group.add_mutually_exclusive_group(required=True)
    vol_mutex_group.add_argument(
        "-v", "--sigma", type=float, metavar='VOL',
        help="Fixed volatility (annualized decimal, e.g., 0.2 for 20%)."
    )
    vol_mutex_group.add_argument(
        "--ticker", type=str, metavar='SYMBOL',
        help="Stock ticker symbol (e.g., AAPL) for ML volatility prediction."
    )

    # --- Simulation Parameters ---
    sim_group = parser.add_argument_group('Monte Carlo Simulation Parameters')
    sim_group.add_argument(
        "-n", "--simulations", type=int, default=10000, metavar='NUM',
        help="Number of Monte Carlo simulation paths."
    )
    sim_group.add_argument(
        "-m", "--steps", type=int, default=100, metavar='NUM',
        help="Number of time steps per simulation path."
    )

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # --- Post-parsing Validation ---
    # Validate fixed sigma if it was provided
    if args.sigma is not None and args.sigma <= 0:
        parser.error("--sigma must be positive.")
    # Validate time to expiration (allow T=0 for intrinsic value calculation)
    if args.time < 0:
         parser.error("--time cannot be negative.")
    # Validate prices are non-negative
    if args.stock < 0:
         parser.error("--stock price cannot be negative.")
    if args.strike < 0:
         parser.error("--strike price cannot be negative.")
    # Validate simulation parameters are positive
    if args.simulations <= 0:
        parser.error("--simulations must be positive.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")

    # Return the validated arguments
    return args

def main():
    """Main function to parse arguments, run pricing models, and print results."""
    start_time = time.time() # Record start time for total execution measurement
    try:
        # Parse command line arguments and perform initial validation
        args = parse_arguments()
    except SystemExit: # Catch argparse errors (like missing required args or validation failures)
        # argparse automatically prints the error message to stderr
        sys.exit(1) # Exit if arguments are invalid

    # Extract parameters from parsed arguments for easier access
    S0 = args.stock
    K = args.strike
    r = args.rate
    T = args.time
    num_simulations = args.simulations
    num_steps = args.steps
    sigma_fixed = args.sigma # This will be None if --ticker was used
    ticker = args.ticker     # This will be None if --sigma was used

    sigma_to_use = None # Variable to hold the final volatility value used for pricing
    vol_source = ""     # String to describe where the volatility value came from ("Fixed" or "ML")

    # --- Determine Volatility Source and Value ---
    if ticker:
        # ML Prediction Path: --ticker was provided by the user
        print(f"\n--- ML Volatility Prediction for {ticker} ---")
        vol_source = f"ML Predicted ({ticker})" # Set description for output
        try:
            # Initialize the volatility predictor class with the specified ticker
            predictor = VolatilityPredictor(ticker=ticker)
            # Call the method to predict volatility.
            # This method should encapsulate fetching data, training (if needed), and predicting.
            sigma_to_use = predictor.predict_volatility()
            print(f"Using predicted volatility: {sigma_to_use*100:.2f}%")
        except (RuntimeError, ValueError, ConnectionError) as e:
            # Handle errors specifically raised by the predictor during its process
            # These could be data fetching errors, training errors, or prediction errors.
            print(f"\nError during volatility prediction: {e}", file=sys.stderr)
            print("Exiting.", file=sys.stderr)
            sys.exit(1) # Exit gracefully on prediction failure
        except Exception as e: # Catch any other unexpected error during prediction/training
             # This catches unforeseen issues in the predictor logic.
             print(f"\nAn unexpected error occurred during volatility prediction/training: {e}", file=sys.stderr)
             sys.exit(1) # Exit gracefully on unexpected failure
    else:
         # Fixed Volatility Path: --sigma was provided (must have a value due to parser logic)
         sigma_to_use = sigma_fixed
         vol_source = "Fixed Input" # Set description for output
         print("\n--- Using Fixed Volatility ---")
         print(f"Provided fixed volatility: {sigma_to_use*100:.2f}%")

    # --- Print Parameters Being Used for Calculation ---
    print("\n--- Option Parameters ---")
    print(f"Underlying Price (S0): {S0:.2f}")
    print(f"Strike Price (K):      {K:.2f}")
    print(f"Risk-free Rate (r):    {r*100:.2f}%")
    print(f"Time to Maturity (T):  {T:.2f} years")
    # Print the volatility value actually used and indicate its source
    print(f"Volatility (sigma):    {sigma_to_use*100:.2f}% ({vol_source})")


    print("\n--- Simulation Parameters ---")
    # Use comma formatting for better readability of large simulation numbers
    print(f"Number of Simulations: {num_simulations:,}")
    print(f"Number of Time Steps:  {num_steps:,}")

    # --- Calculate Option Prices ---
    print("\n--- Calculating Prices ---")
    try:
        # 1. Calculate Black-Scholes price using the determined volatility
        bs_start_time = time.time() # Time the calculation
        bs_price = black_scholes_call(S=S0, K=K, r=r, sigma=sigma_to_use, T=T)
        bs_time = time.time() - bs_start_time
        print(f"Black-Scholes calculation finished in {bs_time:.4f} seconds.")

        # 2. Calculate Monte Carlo price using the determined volatility
        mc_start_time = time.time() # Time the calculation
        # The MC pricing function handles T=0 internally, returning intrinsic value.
        mc_price = monte_carlo_call(
            S0=S0, K=K, r=r, sigma=sigma_to_use, T=T,
            num_steps=num_steps, num_simulations=num_simulations
        )
        mc_time = time.time() - mc_start_time
        if T > 0: # Only print simulation time if a simulation actually ran (T>0)
            print(f"Monte Carlo simulation finished in {mc_time:.4f} seconds.")
        else:
            # If T=0, the MC function returns the intrinsic value without simulating.
            print("Monte Carlo simulation skipped (T=0, price is intrinsic value).")


        # --- Print Results ---
        print("\n--- Results ---")
        # Show results with more precision (e.g., 4 decimal places) for better comparison
        print(f"Analytical Black-Scholes Price: {bs_price:.4f}")
        print(f"Monte Carlo Estimated Price:    {mc_price:.4f}")

        # Calculate and display the difference between the two methods only if T > 0
        # (At T=0, both should yield the same intrinsic value, barring floating point issues)
        if T > 0:
            price_diff = abs(bs_price - mc_price)
            # Calculate percentage difference, handle division by zero if bs_price is 0
            diff_percent = (price_diff / bs_price * 100) if bs_price != 0 else float('inf') if price_diff != 0 else 0.0
            print(f"Difference:                     {price_diff:.4f} ({diff_percent:.2f}%)")


    except ValueError as e:
        # Catch specific ValueErrors raised from pricing functions
        # (e.g., invalid inputs that might have passed initial checks or issues during calculation)
        print(f"\nError during pricing calculation: {e}", file=sys.stderr)
        sys.exit(1) # Exit on calculation error
    except Exception as e:
        # Catch any other unexpected errors during the pricing calculations
        print(f"\nAn unexpected error occurred during pricing: {e}", file=sys.stderr)
        sys.exit(1) # Exit on unexpected error

    # --- End of Script ---
    total_time = time.time() - start_time # Calculate total runtime
    print(f"\nTotal execution time: {total_time:.4f} seconds.")

# This standard Python construct ensures that main() runs only when the script
# is executed directly (e.g., `python src/main.py ...` or `python -m src.main ...`),
# and not when imported as a module into another script.
if __name__ == "__main__":
    main()
