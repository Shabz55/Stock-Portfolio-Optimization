import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Fetch Stock Data
def fetch_stock_data(stocks, start_date, end_date):
    """
    Fetch stock data using yfinance.
    """
    data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']
    return data

# Step 2: Add Moving Averages as Features
def add_moving_averages(data, windows=[5, 10, 20]):
    """
    Add moving averages (SMA and EMA) for each column (stock) in the DataFrame
    using efficient batch processing.
    """
    moving_averages = []  # To store new columns temporarily

    for window in windows:
        for stock in data.columns:  # Process each stock/column individually
            # Calculate SMA and EMA
            sma = data[stock].rolling(window=window).mean()
            ema = data[stock].ewm(span=window, adjust=False).mean()

            # Add new columns to the temporary list
            moving_averages.append(pd.DataFrame({
                f'{stock} {window}-day SMA': sma,
                f'{stock} {window}-day EMA': ema
            }))

    # Concatenate all new moving averages to the original DataFrame
    moving_averages_df = pd.concat(moving_averages, axis=1)
    data = pd.concat([data, moving_averages_df], axis=1)

    return data

# Step 3: Train Linear Regression Model
def train_linear_regression(features, target):
    """
    Train a linear regression model to predict future returns using moving averages.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future returns
    predictions = model.predict(features)
    return model, predictions

# Step 4: Optimize Portfolio Weights
def optimize_portfolio(predicted_returns, cov_matrix):
    """
    Optimize portfolio weights based on predicted returns and covariance matrix.
    """
    num_assets = len(predicted_returns)

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]  # No short selling

    # Objective function: Negative Sharpe Ratio (to maximize Sharpe Ratio)
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, predicted_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility

    # Initial weights (equal allocation)
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Run optimization
    result = minimize(neg_sharpe, initial_weights, bounds=bounds, constraints=constraints)
    return result.x  # Optimal weights

# Step 5: Backtest Portfolio
def backtest_portfolio(stock_returns, optimal_weights):
    """
    Backtest the portfolio using optimal weights.
    """
    portfolio_daily_returns = (stock_returns * optimal_weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns * 100, label="Optimized Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return %")
    plt.title("Portfolio Performance")
    plt.legend()
    plt.show()

# Main Script
if __name__ == "__main__":
    # Define stocks and time period
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA","NVDA"]
    start_date = "2014-11-01"
    end_date = "2024-11-01"

    # Fetch stock data
    stock_data = fetch_stock_data(stocks, start_date, end_date)

    # Add moving averages as features
    windows = [5, 10, 20]
    stock_data = add_moving_averages(stock_data, windows)

    # Create features (moving averages)
    stock_returns = stock_data.pct_change().dropna()

    # Define columns for moving averages
    moving_avg_columns = [f'{stock} {window}-day SMA' for stock in stocks for window in windows] + \
                         [f'{stock} {window}-day EMA' for stock in stocks for window in windows]
    features = stock_data[moving_avg_columns].dropna()

    # Define the prediction horizon (e.g., 5 days)
    prediction_horizon = 20

    # Create the target as the return over the next 'prediction_horizon' days
    target = (stock_data.shift(-prediction_horizon) / stock_data - 1).dropna()

    # Align features and target by dropping any rows with NaN
    features, target = features.align(target, join='inner', axis=0)

    # Drop any remaining rows with NaN in the target
    target = target.dropna()
    features = features.loc[target.index]  # Align features with the cleaned target

    # Train linear regression model and get predictions
    models, predicted_returns = {}, {}
    prediction_results = pd.DataFrame(index=features.index)  # To store predictions and actual returns
    for stock in stocks:
        model, predictions = train_linear_regression(features, target[stock])
        models[stock] = model
        predicted_returns[stock] = predictions

        # Store predictions and actual returns for this stock
        prediction_results[f'{stock} Predicted Return ({prediction_horizon} days)'] = predictions
        prediction_results[f'{stock} Actual Return ({prediction_horizon} days)'] = target[stock]

    # Display predictions and actual returns
    print(f"Predictions vs Actual Returns for {prediction_horizon}-Day Horizon:")
    print(prediction_results.tail(10))  # Display the last 10 rows of predictions and actual returns

    # Convert predicted returns to a DataFrame
    predicted_returns = pd.DataFrame(predicted_returns, index=features.index)

    # Filter the covariance matrix to match the predicted returns
    cov_matrix = stock_returns[predicted_returns.columns].cov()

    # Optimize portfolio weights using predicted returns
    mean_predicted_returns = predicted_returns.mean()
    optimal_weights = optimize_portfolio(mean_predicted_returns, cov_matrix)

    # Display optimal weights
    print("\nOptimal Portfolio Weights:")
    for stock, weight in zip(stocks, optimal_weights):
        print(f"{stock}: {weight:.2%}")

    # Backtest the portfolio
    backtest_portfolio(stock_returns[predicted_returns.columns], optimal_weights)
