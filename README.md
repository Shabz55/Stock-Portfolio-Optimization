# Stock-Portfolio-Optimization
# Stock Portfolio Optimization

This project predicts future stock returns using **Linear Regression** based on historical moving averages and constructs an optimized portfolio to maximize risk-adjusted returns. The model integrates **data scraping**, **feature engineering**, **predictive modeling**, and **portfolio optimization** to build a basic end-to-end investment strategy.

## Features

- Fetches historical stock data via **yfinance**  
- Calculates **SMA** and **EMA** as predictive features  
- Trains **Linear Regression** models to forecast multi-day returns  
- Optimizes portfolio weights using **Sharpe Ratio maximization**  
- Backtests and visualizes portfolio performance over time

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

## Output
- Predicted returns vs. actual returns for selected stocks
- Optimal portfolio weights based on the Sharpe Ratio
- Backtest plot showing cumulative portfolio returns
