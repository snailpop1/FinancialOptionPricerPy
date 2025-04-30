# tests/models/test_black_scholes.py
import pytest
import numpy as np
from scipy.stats import norm
from src.models.black_scholes import european_call_price, calculate_d1_d2

# Define some standard parameters for testing
# These can also be defined as fixtures in conftest.py
S_test = 100.0
K_test = 105.0
r_test = 0.05
sigma_test = 0.2
T_test = 1.0

# Expected values (calculated separately or known good values)
# You might need a financial library or online calculator to verify exact BS prices
expected_price_test = 8.02135 # Example value, verify this!
expected_d1_test = 0.16118 # Example value, verify this!
expected_d2_test = -0.03882 # Example value, verify this!

# Use pytest.approx for comparing floating point numbers
tolerance = 1e-5

def test_calculate_d1_d2_valid():
    """Test d1 and d2 calculation with valid inputs."""
    d1, d2 = calculate_d1_d2(S_test, K_test, r_test, sigma_test, T_test)
    assert d1 == pytest.approx(expected_d1_test, abs=tolerance)
    assert d2 == pytest.approx(expected_d2_test, abs=tolerance)

def test_calculate_d1_d2_invalid_input():
    """Test d1/d2 calculation raises errors for invalid inputs."""
    with pytest.raises(ValueError, match="Volatility.*positive"):
        calculate_d1_d2(S_test, K_test, r_test, 0, T_test) # sigma = 0
    with pytest.raises(ValueError, match="Volatility.*positive"):
        calculate_d1_d2(S_test, K_test, r_test, -0.1, T_test) # sigma < 0
    with pytest.raises(ValueError, match="Time.*positive"):
        calculate_d1_d2(S_test, K_test, r_test, sigma_test, 0) # T = 0
    with pytest.raises(ValueError, match="Time.*positive"):
        calculate_d1_d2(S_test, K_test, r_test, sigma_test, -1.0) # T < 0
    with pytest.raises(ValueError, match="Stock price.*non-negative"):
        calculate_d1_d2(-5, K_test, r_test, sigma_test, T_test) # S < 0
    with pytest.raises(ValueError, match="strike price.*non-negative"):
        calculate_d1_d2(S_test, -10, r_test, sigma_test, T_test) # K < 0


def test_european_call_price_valid():
    """Test Black-Scholes price calculation with valid inputs."""
    price = european_call_price(S_test, K_test, r_test, sigma_test, T_test)
    assert price == pytest.approx(expected_price_test, abs=tolerance)

def test_european_call_price_edge_cases():
    """Test Black-Scholes price calculation for edge cases."""
    # T = 0 (Expiration)
    assert european_call_price(S=110, K=100, r=r_test, sigma=sigma_test, T=0) == pytest.approx(10.0) # ITM
    assert european_call_price(S=100, K=100, r=r_test, sigma=sigma_test, T=0) == pytest.approx(0.0)  # ATM
    assert european_call_price(S=90, K=100, r=r_test, sigma=sigma_test, T=0) == pytest.approx(0.0)   # OTM

    # S = 0 (Stock price is zero)
    assert european_call_price(S=0, K=K_test, r=r_test, sigma=sigma_test, T=T_test) == pytest.approx(0.0)

    # K = 0 (Strike price is zero) - Payoff is S_T, PV is S0
    assert european_call_price(S=S_test, K=0, r=r_test, sigma=sigma_test, T=T_test) == pytest.approx(S_test)


def test_european_call_price_invalid_input():
    """Test Black-Scholes price calculation raises errors for invalid inputs."""
    with pytest.raises(ValueError, match="Volatility.*positive"):
        european_call_price(S_test, K_test, r_test, 0, T_test)
    with pytest.raises(ValueError, match="Time.*negative"):
        european_call_price(S_test, K_test, r_test, sigma_test, -1.0)
    with pytest.raises(ValueError, match="Stock price.*non-negative"):
        european_call_price(-5, K_test, r_test, sigma_test, T_test)
    with pytest.raises(ValueError, match="strike price.*non-negative"):
        european_call_price(S_test, -10, r_test, sigma_test, T_test)