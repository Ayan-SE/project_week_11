import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_forecasted_data(tickers, start_date, end_date):
    """Fetches historical data and creates a DataFrame."""
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

def calculate_returns(df):
    """Calculates daily returns."""
    return df.pct_change().dropna()

def calculate_annual_return(daily_returns):
    """Calculates annual returns from daily returns."""
    return (1 + daily_returns.mean()) ** 252 - 1

def calculate_covariance_matrix(daily_returns):
    """Calculates the covariance matrix."""
    return daily_returns.cov() * 252  # Annualized covariance

def calculate_portfolio_performance(weights, returns, cov_matrix):
    """Calculates portfolio return and volatility."""
    mean_returns = returns.mean().values
    print("Mean returns shape:", mean_returns.shape)  # Debug print
    print("Weights shape:", weights.shape)  # Debug print
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate=0.02):
    """Calculates the Sharpe Ratio."""
    return (portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(returns, cov_matrix):
    """Optimizes portfolio weights to maximize Sharpe Ratio."""
    num_assets = len(returns.columns)
    def neg_sharpe_ratio(weights, returns, cov_matrix):
        ret, vol = calculate_portfolio_performance(weights, returns, cov_matrix)
        return -calculate_sharpe_ratio(ret, vol)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)
    
    optimized_result = minimize(neg_sharpe_ratio, initial_weights, args=(returns, cov_matrix),
                                method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_result.x

def calculate_var(returns, confidence_level=0.95):
    """Calculates Value at Risk (VaR)."""
    return returns.quantile(1 - confidence_level)

def visualize_portfolio_performance(returns, weights, title="Portfolio Performance"):
    """Visualizes cumulative portfolio returns."""
    weighted_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + weighted_returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label="Cumulative Portfolio Returns")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.show()

def risk_return_analysis(returns, cov_matrix, weights):
    """Performs risk-return analysis."""
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, returns, cov_matrix)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility)
    var_tsla = calculate_var(returns['TSLA'])
    
    print(f"Expected Portfolio Return: {portfolio_return:.4f}")
    print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Tesla VaR (95% Confidence): {var_tsla:.4f}")

    # Risk-Return Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(portfolio_volatility, portfolio_return, color='red', label='Optimized Portfolio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.title('Risk-Return Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()