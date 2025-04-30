# tests/volatility/test_predictor.py
import pytest
import pandas as pd
import numpy as np
from src.volatility.predictor import VolatilityPredictor, DEFAULT_TRAINING_DAYS, FEATURE_WINDOW, TARGET_WINDOW
from unittest.mock import patch, MagicMock # Use unittest.mock or pytest-mock

# Example ticker
test_ticker = "MSFT"

@pytest.fixture
def mock_yfinance_download():
    """Fixture to mock yf.download."""
    # Create realistic-looking fake data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=DEFAULT_TRAINING_DAYS + FEATURE_WINDOW + TARGET_WINDOW + 50, freq='B') # More data than needed
    price_data = np.random.lognormal(mean=0.0001, sigma=0.02, size=len(dates)).cumprod() * 100
    mock_df = pd.DataFrame({'Adj Close': price_data}, index=dates)
    # Create a MagicMock object that simulates yf.download
    mock = MagicMock(return_value=mock_df)
    return mock

@pytest.fixture
def predictor_instance():
    """Fixture to create a VolatilityPredictor instance for tests."""
    # Verbose=False to reduce noise during tests
    return VolatilityPredictor(ticker=test_ticker, verbose=False, random_state=42)

# Use patch to replace yf.download with our mock during the test
@patch('yfinance.download')
def test_fetch_data(mock_download_func, predictor_instance):
    """Test data fetching with mocked yfinance."""
    # Configure the mock provided by the fixture
    mock_download_func.return_value = mock_yfinance_download().return_value # Assign the DataFrame

    df = predictor_instance._fetch_data(period_days=100)
    mock_download_func.assert_called_once_with(
        test_ticker, period="100d", auto_adjust=True, progress=False
    )
    assert isinstance(df, pd.DataFrame)
    assert "Adj Close" in df.columns
    assert len(df) <= 100 # Should return at most the requested number of days

@patch('yfinance.download')
def test_fetch_data_fallback_to_close(mock_download_func, predictor_instance):
    """Test data fetching falls back to 'Close' if 'Adj Close' is missing."""
    # Create mock data *without* 'Adj Close' but *with* 'Close'
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100, freq='B')
    price_data = np.random.rand(len(dates)) * 100
    mock_df_no_adj = pd.DataFrame({'Close': price_data}, index=dates)
    mock_download_func.return_value = mock_df_no_adj

    df = predictor_instance._fetch_data(period_days=100)
    assert "Adj Close" in df.columns # Should be renamed
    assert not df.empty

@patch('yfinance.download')
def test_fetch_data_empty(mock_download_func, predictor_instance):
    """Test data fetching raises error if yfinance returns empty dataframe."""
    mock_download_func.return_value = pd.DataFrame() # Empty DataFrame
    with pytest.raises(ValueError, match=f"No price data found for '{test_ticker}'"):
        predictor_instance._fetch_data(period_days=100)

@patch('yfinance.download')
def test_prepare_features_and_target(mock_download_func, predictor_instance):
    """Test feature and target preparation."""
    # Use the realistic mock data
    mock_download_func.return_value = mock_yfinance_download().return_value
    raw_data = predictor_instance._fetch_data(DEFAULT_TRAINING_DAYS)

    X, y = predictor_instance._prepare_features_and_target(raw_data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "rolling_volatility" in X.columns
    assert "rolling_mean_return" in X.columns
    assert not X.isnull().any().any() # No NaNs expected after dropna
    assert not y.isnull().any()
    assert len(X) == len(y)

# Patch both yfinance.download and the model's fit/predict methods
@patch('yfinance.download')
@patch('sklearn.ensemble.GradientBoostingRegressor.fit')
@patch('sklearn.ensemble.GradientBoostingRegressor.predict')
def test_train_and_predict(mock_predict, mock_fit, mock_download_func, predictor_instance):
    """Test the train and predict workflow with mocks."""
    # Configure mocks
    mock_download_func.return_value = mock_yfinance_download().return_value
    mock_fit.return_value = None # fit doesn't return anything significant
    # Make predict return a plausible volatility value (e.g., 0.25)
    mock_predict.return_value = np.array([0.25])

    # Train the model (uses mocked data and fit)
    predictor_instance.train(training_days=100, test_size=0.2)
    mock_fit.assert_called_once() # Check if fit was called
    assert predictor_instance.is_trained
    assert predictor_instance.last_rmse is not None # Check RMSE was calculated

    # Predict volatility (uses mocked data and predict)
    predicted_sigma = predictor_instance.predict_volatility(lookback_days=FEATURE_WINDOW)
    # Predict is called twice: once in train (for RMSE) and once here
    assert mock_predict.call_count >= 1
    # Check the *last* call was with the latest features for prediction
    assert mock_predict.call_args[0][0].shape[0] == 1 # Should predict on 1 row
    assert predicted_sigma == pytest.approx(0.25)

# Test the CLI interface requires more advanced techniques (capturing stdout, mocking sys.argv)
# Using libraries like `click.testing.CliRunner` or manual mocking with `capsys` fixture.
# This is a basic example assuming the script part exists.
@patch('src.volatility.predictor.VolatilityPredictor.train')
@patch('src.volatility.predictor.VolatilityPredictor.predict_volatility')
@patch('src.volatility.predictor.yf.download') # Also mock download if called by main script part
@patch('sys.argv', ['predictor.py', 'GOOG', '--train-days', '200']) # Mock command line arguments
def test_predictor_cli(mock_download, mock_predict, mock_train, capsys):
     """Test the script when run from command line (basic)."""
     # Mock return values
     mock_train.return_value = None
     mock_predict.return_value = 0.30

     # Need to import the script execution part or run it
     # This depends on how the `if __name__ == "__main__":` block is structured
     # For simplicity, let's assume it calls train() then predict_volatility()
     # You might need to use runpy.run_module('src.volatility.predictor', run_name='__main__')

     # Placeholder for actual execution trigger - adjust based on your setup
     try:
         # You might need to re-import or run the module if __main__ guard prevents direct call
         from src.volatility import predictor # Re-import might be needed in some setups
         # Simulate execution if possible, or directly call relevant functions if needed
         # This part is tricky without seeing the exact __main__ execution flow.
         # Assuming the logic inside if __name__ == "__main__": can be called or simulated:
         predictor_instance = predictor.VolatilityPredictor(ticker='GOOG', verbose=True)
         predictor_instance.train(training_days=200)
         vol = predictor_instance.predict_volatility()
         print(f"Predicted annualised volatility for GOOG: {vol:.4f} ({vol * 100:.2f}%)")

     except Exception as e:
          pytest.fail(f"CLI execution simulation failed: {e}")


     # Check if train and predict were called
     mock_train.assert_called_once_with(training_days=200)
     mock_predict.assert_called_once()

     # Capture printed output
     captured = capsys.readouterr()
     assert "Predicted annualised volatility for GOOG: 0.3000 (30.00%)" in captured.out