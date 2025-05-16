# FinancialOptionPricerPy

## Overview
This project implements a Monte Carlo simulation **and** the analytical Black–Scholes formula to estimate the price of a European call option.

**Key Enhancement:** It incorporates Machine-Learning to predict the underlying asset’s volatility from historical price data, providing a more data-driven approach than using a fixed volatility input.

The tool demonstrates the interplay of numerical methods (Monte Carlo), analytical models (Black–Scholes), and machine learning (volatility forecasting) in financial modelling.

## Features
- Prices European call options using Monte Carlo simulation based on **Geometric Brownian Motion (GBM)**.  
- Calculates the analytical price with the **Black–Scholes formula**.  
- **ML Volatility Prediction:** Uses a Gradient-Boosting model (or alternatives such as GARCH or LSTM) trained on historical stock data (`yfinance`) to predict the volatility (σ) needed for pricing.  
- Accepts option parameters and simulation settings via **command-line arguments**.  
- Lets you choose between a fixed volatility or the ML-predicted volatility (by providing a stock ticker).  
- Clear output comparing the Monte Carlo estimate with the Black–Scholes price, indicating the volatility source.  
- **Modular code structure** (`src` directory with sub-packages for models, simulation, and volatility prediction).  

## Theoretical Background

### Geometric Brownian Motion (GBM)
The Monte Carlo simulation assumes that the underlying stock price follows a GBM.  
The stochastic differential equation for the stock price $S_t$ at time $t$ is

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

where  

- $(\mu)$ is the drift rate (we use the risk-free rate \(r\) under risk-neutral pricing);  
- $(\sigma)$ is the volatility of the stock price;  
- $dW_t$ is a Wiener process (random component).

The corresponding discrete-time update (from Itô’s Lemma) is

$$
S_{t+\Delta t} = S_t \,\exp\Bigl[\bigl(r-\tfrac{1}{2}\sigma^{2}\bigr)\Delta t + \sigma\sqrt{\Delta t}\,Z\Bigr],
$$

with $Z \sim \mathcal{N}(0,1)$ and step size $\Delta t$.

### Monte Carlo Simulation for Option Pricing

1. **Simulate Paths** – Generate many (`num_simulations`) price paths from *t = 0* to expiration *T* using small steps  
   Δt = T / num_steps

2. **Terminal Prices** – Record the price *S<sub>T</sub>* at expiration for every path.

3. **Payoffs** – For each path, compute  
   Payoff = max(S<sub>T</sub> − K, 0), where K is the strike price.

4. **Average Payoff** – Average the payoffs across all paths.

5. **Discount** – Present-value the average payoff at rate *r*:  
   Price = exp(−rT) × Average Payoff


### Black–Scholes Formula
The analytical call price is

$$
C(S_t,t) = N(d_1)\,S_t - N(d_2)\,K\,e^{-r(T-t)},
$$

with  

$$
d_1 = \frac{\ln(S_t/K) + \bigl(r + \tfrac12\sigma^2\bigr)(T-t)}{\sigma\sqrt{T-t}},\qquad
d_2 = d_1 - \sigma\sqrt{T-t},
$$

and $N(\cdot)$ the standard-normal CDF.

## ML Volatility Prediction
Volatility $(\sigma)$ is pivotal in option pricing. Instead of assuming a constant value, this project can **predict it**:

1. **Data Acquisition** – Fetch historical daily prices with `yfinance`.  
2. **Feature Engineering** – Compute log-returns, rolling realised volatility, and rolling mean returns (e.g. 60-day window).  
3. **Target Variable** – Future realised volatility, shifted forward.  
4. **Model Training** – Train a regression model (e.g. `GradientBoostingRegressor`) on chronological splits to avoid look-ahead bias.  
5. **Prediction** – Feed the latest features to predict annualised volatility over the option’s life.  
6. **Integration** – Use the predicted σ in both Black–Scholes and Monte Carlo pricing.

## Project Structure
```text
option-pricing-mc/
├── .gitignore         
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── black_scholes.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── monte_carlo.py
│   └── volatility/
│       ├── __init__.py
│       └── predictor.py
├── notebooks/
│   └── simple_volatility_experiment.ipynb # Or volatility_experimentation.ipynb
└── tests/
    ├── __init__.py
    ├── conftest.py       # (Optional) Fixtures for pytest
    ├── test_main.py
    ├── models/
    │   ├── __init__.py
    │   └── test_black_scholes.py
    ├── simulation/
    │   ├── __init__.py
    │   └── test_monte_carlo.py
    └── volatility/
        ├── __init__.py
        └── test_predictor.py
```

## Setup

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd option-pricing-mc 
    ```
    *(Replace `<your-repo-url>` with the actual URL of your repository)*

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

> **Note**
> `requirements.txt` now includes `yfinance`, `scikit-learn`, and `pandas`.

## Usage

Run the simulation from the project root (`option-pricing-mc/`). You have two main ways to specify volatility:

1.  **Using Fixed Volatility**

    Provide the volatility directly with the `--sigma` argument.
    ```bash
    python -m src.main \
        --stock 100 \
        --strike 105 \
        --rate 0.05 \
        --sigma 0.2 \
        --time 1.0 \
        --simulations 10000 \
        --steps 100
    ```

2.  **Using ML-Predicted Volatility**

    Provide the stock ticker with the `--ticker` argument. The script will fetch data, train (if necessary), predict volatility, and use it for pricing. The `--sigma` argument is ignored if `--ticker` is provided.
    ```bash
    # Example using Apple Inc. ticker
    python -m src.main \
        --stock 170 \
        --strike 175 \
        --rate 0.05 \
        --ticker AAPL \
        --time 0.5 \
        --simulations 10000 \
        --steps 100
    ```

> **Tip**
> The `--stock` price should ideally be the current market price for the specified `--ticker` for highest accuracy, although the model uses historical data for volatility prediction.

## Command-Line Arguments

| Argument        | Alias | Description                                                                                                | Required |
| :-------------- | :---- | :--------------------------------------------------------------------------------------------------------- | :------- |
| `--stock`       | `-s`  | Current stock price (e.g., `170.5`)                                                                        | ✅        |
| `--strike`      | `-k`  | Option strike price (e.g., `175`)                                                                          | ✅        |
| `--rate`        | `-r`  | Risk-free interest rate (annualised, e.g., `0.05` for 5%)                                                  | ✅        |
| `--time`        | `-t`  | Time to expiration in years (e.g., `0.5` for 6 months)                                                     | ✅        |
| `--sigma`       | `-v`  | Fixed volatility (annualised, e.g., `0.25`). Required if `--ticker` is **not** used.                       | ⚠️        |
| `--ticker`      | `-`   | Stock ticker (e.g., `AAPL`, `GOOGL`). If provided, ML prediction is used for volatility and `--sigma` is ignored. Required if `--sigma` is **not** used. | ⚠️        |
| `--simulations` | `-n`  | Number of Monte Carlo simulation paths (default: `10000`)                                                  | -        |
| `--steps`       | `-m`  | Number of time steps in each simulation path (default: `100`)                                              | -        |


## Example Output (Using ML Volatility)

```text
--- ML Volatility Prediction for AAPL ---
Fetching historical data for AAPL...
Successfully fetched 755 data points.
Preparing features...
Feature preparation complete. Shape: (694, 2)
Training model on 555 samples...
Model training complete.
Model Evaluation (Test RMSE): 0.03XXX
Predicting future volatility...
Fetching historical data for AAPL...
Successfully fetched 65 data points.
Using latest features for prediction:
            rolling_volatility  rolling_mean_return
Date                                               
YYYY-MM-DD              0.27XX              -0.05XX
Raw Predicted Annualized Volatility: 0.28XX
Using predicted volatility: 28.XX%

--- Option Parameters ---
Underlying Price (S0): 170.00
Strike Price (K):      175.00
Risk-free Rate (r):    5.00%
Volatility (sigma):    28.XX% (ML Predicted (AAPL))
Time to Maturity (T):  0.50 years

--- Simulation Parameters ---
Number of Simulations: 10,000
Number of Time Steps:  100

--- Calculating Prices ---
Black-Scholes calculation finished in 0.XXX seconds.
Monte Carlo simulation finished in X.XXX seconds.

--- Results ---
Analytical Black-Scholes Price: X.XXX
Monte Carlo Estimated Price:    Y.YYY
Difference:                     Z.ZZZ (P.PP%)

Total execution time: Y.YYY seconds.

```
## Table of Contents

* [FinancialOptionPricerPy](#financialoptionpricerpy)
* [Overview](#overview)
* [Features](#features)
* [Theoretical Background](#theoretical-background)
    * [Geometric Brownian Motion (GBM)](#geometric-brownian-motion-gbm)
    * [Monte Carlo Simulation for Option Pricing](#monte-carlo-simulation-for-option-pricing)
    * [Black–Scholes Formula](#blackscholes-formula)
    * [ML Volatility Prediction](#ml-volatility-prediction)
* [Project Structure](#project-structure)
* [Setup](#setup)
* [Usage](#usage)
    * [Using Fixed Volatility](#using-fixed-volatility)  
    * [Using ML-Predicted Volatility](#using-ml-predicted-volatility) 
* [Command-Line Arguments](#command-line-arguments)
* [Example Output (Using ML Volatility)](#example-output-using-ml-volatility)
