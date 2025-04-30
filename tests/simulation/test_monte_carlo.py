# tests/simulation/test_monte_carlo.py
import pytest
import numpy as np
from src.simulation.monte_carlo import monte_carlo_european_call_price, simulate_gbm_paths

# Define standard parameters (can be fixtures)
S0_test = 100.0
K_test = 105.0
r_test = 0.05
sigma_test = 0.2
T_test = 1.0
num_steps_test = 50 # Fewer steps/sims for faster tests
num_simulations_test = 1000 # Fewer steps/sims for faster tests

# Expected Black-Scholes price for comparison (from previous test)
bs_price_ref = 8.02135
mc_tolerance = 1.0 # Monte Carlo has inherent randomness, allow larger tolerance

@pytest.fixture(scope="module")
def mocked_rng():
    """Fixture to provide a deterministic random number generator for tests."""
    # Seed the generator for reproducible results during testing
    return np.random.default_rng(42) # Use a fixed seed

def test_simulate_gbm_paths_output_shape(mocked_rng):
    """Test the shape and start value of simulated paths."""
    # Temporarily replace the default_rng used in the function
    # This requires modifying simulate_gbm_paths or using a mocking library like pytest-mock
    # For simplicity, we'll assume mocked_rng can be passed or monkeypatched if needed.
    # Here, we'll just test the shape without full mocking for brevity.
    # NOTE: Proper testing would involve mocking np.random.default_rng()
    paths, dt = simulate_gbm_paths(S0_test, r_test, sigma_test, T_test, num_steps_test, num_simulations_test)

    assert paths.shape == (num_simulations_test, num_steps_test + 1)
    assert np.all(paths[:, 0] == S0_test) # Check initial price
    assert dt == pytest.approx(T_test / num_steps_test)

def test_simulate_gbm_paths_invalid_input():
    """Test GBM simulation raises errors for invalid inputs."""
    with pytest.raises(ValueError, match="Initial stock price.*positive"):
        simulate_gbm_paths(0, r_test, sigma_test, T_test, num_steps_test, num_simulations_test)
    with pytest.raises(ValueError, match="Volatility.*positive"):
        simulate_gbm_paths(S0_test, r_test, 0, T_test, num_steps_test, num_simulations_test)
    with pytest.raises(ValueError, match="Time.*positive"):
        simulate_gbm_paths(S0_test, r_test, sigma_test, 0, num_steps_test, num_simulations_test)
    with pytest.raises(ValueError, match="steps.*positive"):
        simulate_gbm_paths(S0_test, r_test, sigma_test, T_test, 0, num_simulations_test)
    with pytest.raises(ValueError, match="simulations.*positive"):
        simulate_gbm_paths(S0_test, r_test, sigma_test, T_test, num_steps_test, 0)

def test_monte_carlo_call_price_valid():
    """Test Monte Carlo price estimation against Black-Scholes (with tolerance)."""
    # Note: This test might be flaky due to randomness.
    # Running multiple times or using more simulations helps.
    # Mocking random numbers is the most robust way.
    mc_price = monte_carlo_european_call_price(
        S0_test, K_test, r_test, sigma_test, T_test, num_steps_test, num_simulations_test
    )
    # Check if MC price is reasonably close to BS price
    assert mc_price == pytest.approx(bs_price_ref, abs=mc_tolerance)

def test_monte_carlo_call_price_edge_case_T0():
    """Test Monte Carlo price for T=0 (should return intrinsic value)."""
    assert monte_carlo_european_call_price(
        S0=110, K=100, r=r_test, sigma=sigma_test, T=0, num_steps=10, num_simulations=100
    ) == pytest.approx(10.0) # ITM
    assert monte_carlo_european_call_price(
        S0=90, K=100, r=r_test, sigma=sigma_test, T=0, num_steps=10, num_simulations=100
    ) == pytest.approx(0.0) # OTM

def test_monte_carlo_call_price_invalid_input():
    """Test Monte Carlo price calculation raises errors for invalid inputs."""
    with pytest.raises(ValueError, match="Strike price.*non-negative"):
        monte_carlo_european_call_price(S0_test, -10, r_test, sigma_test, T_test, num_steps_test, num_simulations_test)
    with pytest.raises(ValueError, match="Time.*negative"):
         monte_carlo_european_call_price(S0_test, K_test, r_test, sigma_test, -1.0, num_steps_test, num_simulations_test)
    # Other input errors are caught by simulate_gbm_paths tests