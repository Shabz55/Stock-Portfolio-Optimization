# Stock-Portfolio-Optimization

This project builds a complete machine-learning-based portfolio optimizer. It downloads historical stock data, engineers technical indicators, predicts multi-day forward returns using a Random Forest model, optimizes portfolio weights using Sharpe ratio maximization, and backtests the resulting portfolio.

---

## Overview

This script performs the following:

- Fetches historical stock data from Yahoo Finance  
- Generates technical indicators (SMA, EMA, returns, volatility)  
- Predicts future returns using a Random Forest Regressor  
- Calculates error metrics for each stock  
- Optimizes portfolio weights based on predicted returns  
- Backtests the optimized portfolio  
- Visualizes predicted vs. actual returns and cumulative performance  

All functionality is contained in `Portfolio_optimization.py`.

---

## Features

1. **Data Collection**
   - Auto-adjusted daily close prices using yfinance
   - User-specified date range and stock tickers

2. **Feature Engineering**
   - SMA and EMA for 5, 10, and 20-day windows
   - 1D, 5D, 20D returns
   - 20-day rolling volatility

3. **Machine Learning**
   - Random Forest Regressor (300 estimators, max depth 6)
   - Predicts horizon-day forward returns
   - Calculates MAE for each stock
   - Creates a grid plot of predicted vs actual returns

4. **Portfolio Optimization**
   - Maximizes Sharpe ratio using `scipy.optimize.minimize`
   - No short selling (weights between 0 and 1)
   - Weights sum to 1 constraint

5. **Backtesting**
   - Computes cumulative portfolio returns
   - Plots portfolio performance

---

## Technologies Used

- Python  
- pandas, NumPy  
- scikit-learn  
- yfinance  
- scipy (for optimization)  
- matplotlib (for plotting)

## Key Impact

This project showcases practical skills in **data-driven finance**, **machine learning**, and **portfolio management**, providing a hands-on demonstration of how to apply Python to real-world investment strategies.

## How to Run

1. Clone this repository  
2. Install required libraries:  
   ```bash
   pip install yfinance pandas numpy scikit-learn scipy matplotlib
3. Run command (changable tickers and time range)
   ```bash
   python Portfolio_optimization.py AAPL AMZN GOOGL --start 2016-01-01 --end 2024-12-31

## Output
- Predicted returns vs. actual returns for selected stocks
- Optimal portfolio weights based on the Sharpe Ratio
- Backtest plot showing cumulative portfolio returns


