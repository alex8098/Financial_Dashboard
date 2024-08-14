import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.option_metrics import calculate_option_metrics, prob_profit

def run():
    st.title("Options Analysis")
    
    # User inputs for option parameters
    st.sidebar.header("Option Parameters")
    S = st.sidebar.number_input("Current Stock Price", value=100.0)
    K = st.sidebar.number_input("Strike Price", value=100.0)
    T = st.sidebar.number_input("Time to Expiration (in years)", value=1.0, min_value=0.0)
    r = st.sidebar.number_input("Risk-free Rate", value=0.05)
    sigma = st.sidebar.number_input("Volatility", value=0.2)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

    # Calculate option metrics
    option_price, delta, gamma, theta, vega = calculate_option_metrics(S, K, T, r, sigma, option_type)

    # Display results
    st.subheader("Option Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Option Price", f"${option_price:.2f}")
        st.metric("Delta", f"{delta:.4f}")
        st.metric("Gamma", f"{gamma:.4f}")
    with col2:
        st.metric("Theta", f"{theta:.4f}")
        st.metric("Vega", f"{vega:.4f}")

    # Advanced Visualizations
    st.subheader("Advanced Option Analysis Plots")

    # Option Price vs. Stock Price for different times to expiration
    st.write("### Option Price vs. Stock Price for Different Times to Expiration")
    stock_prices = np.linspace(max(0, S - 50), S + 50, 100)
    times = [0.25, 0.5, 1, 2]  # 3 months, 6 months, 1 year, 2 years
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in times:
        option_prices = [calculate_option_metrics(s, K, t, r, sigma, option_type)[0] for s in stock_prices]
        ax.plot(stock_prices, option_prices, label=f'T = {t} year(s)')
    
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Option Price")
    ax.set_title(f"{option_type.capitalize()} Option Price vs. Stock Price")
    ax.legend()
    st.pyplot(fig)

    # Greeks vs. Stock Price
    st.write("### Greeks vs. Stock Price")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for ax, greek in zip(axes, ['Delta', 'Gamma', 'Theta', 'Vega']):
        for t in times:
            greek_values = [calculate_option_metrics(s, K, t, r, sigma, option_type)[['Delta', 'Gamma', 'Theta', 'Vega'].index(greek) + 1] for s in stock_prices]
            ax.plot(stock_prices, greek_values, label=f'T = {t} year(s)')
        ax.set_xlabel("Stock Price")
        ax.set_ylabel(greek)
        ax.set_title(f"{greek} vs. Stock Price")
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

    # Probability of Profit
    st.write("### Probability of Profit")
    stock_prices = np.linspace(max(0, S - 50), S + 50, 100)
    prob_profits = [prob_profit(s, K, T, r, sigma, option_type) for s in stock_prices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_prices, prob_profits)
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Probability of Profit")
    ax.set_title(f"Probability of Profit for {option_type.capitalize()} Option")
    ax.axhline(y=0.5, color='r', linestyle='--')
    ax.axvline(x=K, color='g', linestyle='--')
    st.pyplot(fig)