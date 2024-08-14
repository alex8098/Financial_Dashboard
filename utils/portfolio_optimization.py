import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def portfolio_sharpe_ratio(weights, returns, rf=0.02):
    p_ret = portfolio_return(weights, returns)
    p_vol = portfolio_volatility(weights, returns)
    return (p_ret - rf) / p_vol

def neg_sharpe_ratio(weights, returns, rf=0.02):
    return -portfolio_sharpe_ratio(weights, returns, rf)

def portfolio_metrics(returns):
    return pd.DataFrame({
        'Annual Return': returns.mean() * 252,
        'Annual Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    })

def efficient_frontier(returns, num_portfolios=1000):
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_ret = portfolio_return(weights, returns)
        portfolio_vol = portfolio_volatility(weights, returns)
        results[0,i] = portfolio_ret
        results[1,i] = portfolio_vol
        results[2,i] = portfolio_ret / portfolio_vol

    # Optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    optimized = minimize(neg_sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = optimized.x
    opt_return = portfolio_return(opt_weights, returns)
    opt_volatility = portfolio_volatility(opt_weights, returns)
    opt_sharpe = portfolio_sharpe_ratio(opt_weights, returns)

    return opt_weights, opt_return, opt_volatility, opt_sharpe, results, weights_record

def monte_carlo_simulation(returns, weights, num_simulations=1000, num_days=252, initial_investment=10000):
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (num_days, num_simulations))
    portfolio_returns = np.dot(simulated_returns, weights)
    portfolio_values = initial_investment * (1 + portfolio_returns).cumprod(axis=0)
    return portfolio_values