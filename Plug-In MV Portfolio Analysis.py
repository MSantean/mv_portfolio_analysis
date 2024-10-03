# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:02:58 2024

@author: Mihail
"""

import yfinance as yf
import pandas as pd
import numpy as np
import math
from numpy.linalg import inv
from datetime import date
import matplotlib.pyplot as plt

# Helper functions
def close_to_returns(series):
    return (series[1:] - series[:-1]) / series[:-1]

def get_returns(tickers, N):
    data = pd.DataFrame(yf.download(tickers, start="1980-12-12", end=date.today()).iloc[:, : len(tickers.split())]).tail(2*N+1)
    data = data.to_numpy().T
    R = [close_to_returns(data[i]) for i in range(len(data))]
    return np.array(R)

def estimate_mean(R):
    return np.mean(R, axis=1)

def estimate_covariance(R):
    return np.cov(R)

def optimal_w(gamma, mu_hat, Sigma_hat):
    dim = len(mu_hat)
    inv_Sigma = inv(Sigma_hat)
    B = np.dot(np.dot(mu_hat.T, inv_Sigma), np.ones(dim).T)
    C = np.dot(np.dot(np.ones(dim).T, inv_Sigma), np.ones(dim))
    return (1 / gamma) * np.dot(inv_Sigma, mu_hat) - ((B - gamma) / (gamma * C)) * np.dot(inv_Sigma, np.ones(dim))

# Generate and display empirical statistics
def display_statistics(returns, portfolio_name, rf):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = (mean_return - rf) / std_dev
    print(f"\n{portfolio_name} Portfolio Statistics:")
    print(f"Mean Return: {mean_return:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# Main function implementing rolling window
def rolling_backtest(R, N, gamma, rf):
    num_assets = R.shape[0]
    num_days = R.shape[1]
    
    realized_returns_mv = []  # To store realized returns for mean-variance portfolio
    realized_returns_equal = []  # To store realized returns for 1/N portfolio

    # Loop over each window
    for i in range(num_days - N):
        # Define in-sample period (rolling window)
        in_sample_data = R[:, i:i+N]

        # Estimate parameters and compute optimal weights
        mu_hat = estimate_mean(in_sample_data)
        Sigma_hat = estimate_covariance(in_sample_data)
        w_mv = optimal_w(gamma, mu_hat, Sigma_hat)
        w_equal = np.ones(num_assets) / num_assets  # 1/N portfolio

        # Out-of-sample return is on the day after the in-sample period
        next_day_return = R[:, i+N]  # Day i+N is the next day after in-sample

        # Calculate realized returns for both portfolios
        realized_returns_mv.append(np.dot(w_mv, next_day_return))
        realized_returns_equal.append(np.dot(w_equal, next_day_return))

    # Display the statistics for both portfolios
    display_statistics(realized_returns_mv, "Mean-Variance", rf)
    display_statistics(realized_returns_equal, "1/N", rf)

# Parameters
tickers = 'AAPL AMD AMZN GOOG META MSFT NVDA TSLA'
N = 126  # Window size
gamma = 5.5
rf = 0.0379 / 365  # Daily risk-free rate

# Get returns for the entire period
R = get_returns(tickers, N)

# Run the rolling backtest
rolling_backtest(R, N, gamma, rf)
