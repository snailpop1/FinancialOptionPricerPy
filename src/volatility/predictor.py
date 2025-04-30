# src/volatility/predictor.py
"""Volatility prediction module with optional quiet mode
---------------------------------------------------------
Downloads historical price data via *yfinance*, engineers a couple of
rolling‑statistics features and trains a ``GradientBoostingRegressor`` to
forecast **future annualised volatility**.  All console chatter can now
be silenced by setting ``verbose=False`` or by supplying ``--quiet`` when
using the CLI.

Highlights
~~~~~~~~~~
* Robust column handling – works whether *yfinance* returns ``Adj Close``
  or just ``Close``.
* Compatible with older scikit‑learn versions (doesn’t rely on the
  ``squared=`` kwarg in :pyfunc:`sklearn.metrics.mean_squared_error`).
* Flexible logging – informational messages are routed through the
  :pymod:`logging` library and respect the *verbose* flag.
"""

from __future__ import annotations

import logging
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("volatility.predictor")

# ────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_TRAINING_DAYS: int = 3* 365        # 1 year of history for training
FEATURE_WINDOW: int = 30                # 30‑day rolling features
TARGET_WINDOW: int = 5                  # predict 5‑day realised vol
RANDOM_STATE: int = 42


class VolatilityPredictor:
    """Predict future (annualised) volatility for a single equity."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, ticker: str, *, verbose: bool = True, random_state: int = RANDOM_STATE):
        self.ticker: str = ticker.upper()
        self.verbose: bool = verbose
        self.model: GradientBoostingRegressor = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            loss="squared_error",
            random_state=random_state,
        )
        self.is_trained: bool = False
        self.last_rmse: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str, level: int = logging.INFO) -> None:  # centralised logging
        if self.verbose:
            logger.log(level, msg)

    def _fetch_data(self, period_days: int) -> pd.DataFrame:
        """Download *period_days* of price data and return a DF with a
        single column named ``Adj Close``. Falls back to ``Close`` if the
        adjusted column is missing (this is common when *auto_adjust* is
        set to *True*).
        """
        df = yf.download(
            self.ticker,
            period=f"{period_days}d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No price data found for '{self.ticker}'.")

        if "Adj Close" in df.columns:
            price_df = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            self._log("'Adj Close' not found – using 'Close' column instead.")
            price_df = df[["Close"]].rename(columns={"Close": "Adj Close"})
        else:
            raise ValueError(
                f"Neither 'Adj Close' nor 'Close' found for '{self.ticker}'."
            )

        price_df.dropna(inplace=True)
        return price_df.tail(period_days)

    def _prepare_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineer features and build the prediction target."""
        df = data.copy()
        df["log_return"] = np.log(df["Adj Close"].div(df["Adj Close"].shift(1)))

        df["rolling_volatility"] = (
            df["log_return"]
            .rolling(FEATURE_WINDOW, min_periods=int(FEATURE_WINDOW * 0.8))
            .std(ddof=0)
            * np.sqrt(252)
        )
        df["rolling_mean_return"] = (
            df["log_return"]
            .rolling(FEATURE_WINDOW, min_periods=int(FEATURE_WINDOW * 0.8))
            .mean()
            * 252
        )

        df["future_realised_volatility"] = (
            df["log_return"]
            .rolling(TARGET_WINDOW, min_periods=int(TARGET_WINDOW * 0.8))
            .std(ddof=0)
            .shift(-TARGET_WINDOW)
            * np.sqrt(252)
        )

        df.dropna(inplace=True)
        X = df[["rolling_volatility", "rolling_mean_return"]]
        y = df["future_realised_volatility"]

        if X.empty or y.empty:
            raise ValueError("No data left after feature engineering – check window sizes.")
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, *, training_days: int = DEFAULT_TRAINING_DAYS, test_size: float = 0.2) -> None:
        """Fetch data → engineer features → train & evaluate the model."""
        raw = self._fetch_data(training_days)
        X, y = self._prepare_features_and_target(raw)

        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.last_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        self.is_trained = True
        self._log(f"Training finished. RMSE = {self.last_rmse:.4f}")

    def predict_volatility(self, *, lookback_days: int = FEATURE_WINDOW) -> float:
        """Return the predicted **annualised σ** for the next
        :data:`TARGET_WINDOW` days."""
        if not self.is_trained:
            self._log("Model not trained – training automatically now…", logging.DEBUG)
            self.train()

        raw = self._fetch_data(lookback_days + FEATURE_WINDOW + TARGET_WINDOW)
        raw["log_return"] = np.log(raw["Adj Close"].div(raw["Adj Close"].shift(1)))
        raw["rolling_volatility"] = (
            raw["log_return"]
            .rolling(FEATURE_WINDOW, min_periods=int(FEATURE_WINDOW * 0.8))
            .std(ddof=0)
            * np.sqrt(252)
        )
        raw["rolling_mean_return"] = (
            raw["log_return"]
            .rolling(FEATURE_WINDOW, min_periods=int(FEATURE_WINDOW * 0.8))
            .mean()
            * 252
        )
        raw.dropna(inplace=True)
        if raw.empty:
            raise ValueError("Insufficient data after feature construction – cannot predict.")

        latest = raw[["rolling_volatility", "rolling_mean_return"]].iloc[[-1]]
        sigma = float(self.model.predict(latest)[0])

        if sigma <= 0:  # sanity‑check
            self._log("Model predicted non‑positive vol – using empirical value.", logging.WARNING)
            sigma = float(raw["rolling_volatility"].iloc[-1])
        return sigma


# ────────────────────────────────────────────────────────────────────────────
# CLI usage
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict annualised volatility with Gradient Boosting.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--train-days", type=int, default=DEFAULT_TRAINING_DAYS, help="How many past days to use for training")
    parser.add_argument("--quiet", action="store_true", help="Suppress informational output")
    args = parser.parse_args()

    predictor = VolatilityPredictor(args.ticker, verbose=not args.quiet)
    predictor.train(training_days=args.train_days)
    vol = predictor.predict_volatility()

    print(f"Predicted annualised volatility for {args.ticker.upper()}: {vol:.4f} ({vol * 100:.2f}%)")
