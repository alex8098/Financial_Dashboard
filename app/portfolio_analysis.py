import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
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
    return results, weights_record

def run():
    st.title("Advanced Portfolio Analysis and Optimization")

    # User inputs
    st.sidebar.header("Portfolio Parameters")
    tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN").split(',')
    start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=365*5))
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now())

    # Fetch data
    @st.cache_data
    def load_data(tickers, start, end):
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        return data

    data = load_data(tickers, start_date, end_date)

    # Calculate returns
    returns = data.pct_change().dropna()

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Calculate and display risk-return metrics
    st.subheader("Risk-Return Metrics")
    metrics = pd.DataFrame({
        'Annual Return': returns.mean() * 252,
        'Annual Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    })
    st.dataframe(metrics)

    # Portfolio optimization
    st.subheader("Portfolio Optimization")

    num_assets = len(tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)

    optimized = minimize(neg_sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    st.write("Optimized Portfolio Weights:")
    opt_weights = pd.DataFrame(optimized.x, index=tickers, columns=['Weight'])
    st.dataframe(opt_weights)

    opt_return = portfolio_return(optimized.x, returns)
    opt_volatility = portfolio_volatility(optimized.x, returns)
    opt_sharpe = portfolio_sharpe_ratio(optimized.x, returns)

    st.write(f"Optimized Portfolio Metrics:")
    st.write(f"Expected Annual Return: {opt_return:.2%}")
    st.write(f"Expected Annual Volatility: {opt_volatility:.2%}")
    st.write(f"Sharpe Ratio: {opt_sharpe:.2f}")

    # Efficient Frontier
    st.subheader("Efficient Frontier")

    results, _ = efficient_frontier(returns)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_title('Efficient Frontier')
    ax.scatter(opt_volatility, opt_return, c='red', s=50)  # Highlight the optimized portfolio
    st.pyplot(fig)

    # Monte Carlo Simulation
    st.subheader("Monte Carlo Simulation")

    num_simulations = 1000
    num_days = 252

    def monte_carlo_simulation(returns, weights, initial_investment=10000):
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (num_days, num_simulations))
        portfolio_returns = np.dot(simulated_returns, weights)
        portfolio_values = initial_investment * (1 + portfolio_returns).cumprod(axis=0)
        return portfolio_values

    simulated_values = monte_carlo_simulation(returns, optimized.x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(simulated_values)
    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Monte Carlo Simulation of Optimized Portfolio')
    st.pyplot(fig)

    final_values = simulated_values[-1, :]
    var_95 = np.percentile(final_values, 5)
    var_99 = np.percentile(final_values, 1)

    st.write(f"Value at Risk (95% confidence): ${10000 - var_95:.2f}")
    st.write(f"Value at Risk (99% confidence): ${10000 - var_99:.2f}")