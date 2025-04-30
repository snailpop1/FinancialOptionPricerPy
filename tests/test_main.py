# tests/test_main.py
import pytest
from unittest.mock import patch, MagicMock
import sys

# Mock the functions imported by main.py BEFORE importing main itself
# This ensures main.py gets the mocked versions when it's imported
mock_bs = MagicMock(return_value=8.0) # Mock Black-Scholes function
mock_mc = MagicMock(return_value=8.1) # Mock Monte Carlo function
mock_predictor_instance = MagicMock() # Mock the predictor instance
mock_predictor_instance.predict_volatility.return_value = 0.25 # Mock its predict method
mock_volatility_predictor_class = MagicMock(return_value=mock_predictor_instance) # Mock the class itself

# Apply patches using decorators or context managers before importing or running main
@patch('src.models.black_scholes.european_call_price', mock_bs)
@patch('src.simulation.monte_carlo.monte_carlo_european_call_price', mock_mc)
@patch('src.volatility.predictor.VolatilityPredictor', mock_volatility_predictor_class)
def test_main_fixed_volatility(capsys):
    """Test main() with fixed volatility arguments."""
    test_args = [
        'src/main.py', # Script name
        '-s', '100',   # Stock price
        '-k', '105',   # Strike price
        '-r', '0.05',  # Rate
        '-t', '1.0',   # Time
        '-v', '0.2',   # Fixed volatility
        '--simulations', '500', # Fewer sims for test
        '--steps', '20'         # Fewer steps for test
    ]
    # Use patch.object to temporarily replace sys.argv
    with patch.object(sys, 'argv', test_args):
        from src import main # Import main *after* patches are applied
        main.main() # Run the main function

    # Check if the mocked functions were called correctly
    mock_bs.assert_called_once_with(S=100.0, K=105.0, r=0.05, sigma=0.2, T=1.0)
    mock_mc.assert_called_once_with(S0=100.0, K=105.0, r=0.05, sigma=0.2, T=1.0, num_steps=20, num_simulations=500)
    mock_volatility_predictor_class.assert_not_called() # Predictor should not be called

    # Check the output captured by capsys
    captured = capsys.readouterr()
    assert "Using Fixed Volatility" in captured.out
    assert "Provided fixed volatility: 20.00%" in captured.out
    assert "Analytical Black-Scholes Price: 8.0000" in captured.out # From mock_bs return
    assert "Monte Carlo Estimated Price:    8.1000" in captured.out # From mock_mc return
    assert "Difference:" in captured.out # Check difference is calculated


# Reset mocks for the next test
@patch('src.models.black_scholes.european_call_price', mock_bs)
@patch('src.simulation.monte_carlo.monte_carlo_european_call_price', mock_mc)
@patch('src.volatility.predictor.VolatilityPredictor', mock_volatility_predictor_class)
def test_main_ml_volatility(capsys):
    """Test main() with ML volatility arguments."""
    # Reset call counts on mocks before the test
    mock_bs.reset_mock()
    mock_mc.reset_mock()
    mock_volatility_predictor_class.reset_mock()
    mock_predictor_instance.predict_volatility.reset_mock()


    test_args = [
        'src/main.py',
        '-s', '100',
        '-k', '105',
        '-r', '0.05',
        '-t', '1.0',
        '--ticker', 'AAPL', # Use ticker for ML volatility
        '--simulations', '500',
        '--steps', '20'
    ]
    with patch.object(sys, 'argv', test_args):
        # Need to ensure main is reloaded or imported correctly if run previously
        # Using importlib.reload might be necessary in complex scenarios
        from src import main
        import importlib
        importlib.reload(main) # Reload to ensure mocks are fresh if needed

        main.main()

    # Check if predictor was called and its result used
    mock_volatility_predictor_class.assert_called_once_with(ticker='AAPL')
    mock_predictor_instance.predict_volatility.assert_called_once()
    predicted_vol = mock_predictor_instance.predict_volatility.return_value # Should be 0.25

    # Check BS and MC were called with the *predicted* volatility
    mock_bs.assert_called_once_with(S=100.0, K=105.0, r=0.05, sigma=predicted_vol, T=1.0)
    mock_mc.assert_called_once_with(S0=100.0, K=105.0, r=0.05, sigma=predicted_vol, T=1.0, num_steps=20, num_simulations=500)

    # Check output
    captured = capsys.readouterr()
    assert "ML Volatility Prediction for AAPL" in captured.out
    assert f"Using predicted volatility: {predicted_vol*100:.2f}%" in captured.out
    assert f"Volatility (sigma):    {predicted_vol*100:.2f}% (ML Predicted (AAPL))" in captured.out
    assert "Analytical Black-Scholes Price: 8.0000" in captured.out
    assert "Monte Carlo Estimated Price:    8.1000" in captured.out

def test_argument_parsing_errors():
    """Test that argument parsing catches errors (requires calling parse_arguments directly)."""
    # Need to import parse_arguments if accessible, or test via main's SystemExit
    from src.main import parse_arguments

    with pytest.raises(SystemExit): # argparse raises SystemExit on error
         with patch.object(sys, 'argv', ['src/main.py', '-s', '100']): # Missing required args
             parse_arguments()

    with pytest.raises(SystemExit):
         with patch.object(sys, 'argv', ['src/main.py', '-s', '100', '-k', '100', '-r', '0.05', '-t', '1', '-v', '-0.1']): # Invalid sigma
             parse_arguments()

    with pytest.raises(SystemExit):
         with patch.object(sys, 'argv', ['src/main.py', '-s', '100', '-k', '100', '-r', '0.05', '-t', '1']): # Missing volatility source
             parse_arguments()

    with pytest.raises(SystemExit):
         with patch.object(sys, 'argv', ['src/main.py', '-s', '100', '-k', '100', '-r', '0.05', '-t', '1', '-v', '0.2', '--ticker', 'MSFT']): # Both volatility sources
             parse_arguments()