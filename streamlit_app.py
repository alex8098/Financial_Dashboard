import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Advanced Quant Analysis App", layout="wide")

# Function to calculate option metrics
def calculate_option_metrics(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
    else:  # put option
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return option_price, delta, gamma, theta, vega

# Sidebar for page selection
page = st.sidebar.selectbox("Choose a page", ["Stock Analysis", "Options Analysis", "Portfolio Analysis", "Machine Learning Playground"])

if page == "Stock Analysis":
    # Sidebar for user input
    st.sidebar.header("User Input")
    ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())

    # Fetch data
    @st.cache_data
    def load_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        return data

    data = load_data(ticker, start_date, end_date)

    # Main page
    st.title(f"Advanced Quantitative Analysis Dashboard - {ticker}")

    # Display raw data
    st.subheader("Raw Data")
    st.write(data)

    # Plot stock prices
    st.subheader("Stock Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['Open'], label='Open Price')
    ax.fill_between(data.index, data['High'], data['Low'], alpha=0.3, label='High-Low Range')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} Stock Price")
    ax.legend()
    st.pyplot(fig)

    # Calculate and display basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()

    # Plot daily returns distribution
    st.subheader("Daily Returns Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data['Daily Return'].dropna(), kde=True, ax=ax)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{ticker} Daily Returns Distribution")
    st.pyplot(fig)

    # Calculate and plot moving averages
    st.subheader("Moving Averages")
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['MA50'], label='50-day MA')
    ax.plot(data.index, data['MA200'], label='200-day MA')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} Moving Averages")
    ax.legend()
    st.pyplot(fig)

    # Perform time series decomposition
    st.subheader("Time Series Decomposition")
    decomposition = seasonal_decompose(data['Close'], model='additive', period=30)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title("Observed")
    decomposition.trend.plot(ax=ax2)
    ax2.set_title("Trend")
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title("Seasonal")
    decomposition.resid.plot(ax=ax4)
    ax4.set_title("Residual")
    plt.tight_layout()
    st.pyplot(fig)

    # Simple price prediction
    st.subheader("Simple Price Prediction (Linear Regression)")

    # Prepare data for prediction
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    X = np.array(range(len(scaled_data))).reshape(-1, 1)
    y = scaled_data

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Price')
    ax.plot(X_test, y_pred, color='red', label='Predicted Price')
    ax.set_xlabel("Time")
    ax.set_ylabel("Scaled Price")
    ax.set_title(f"{ticker} Price Prediction")
    ax.legend()
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(f"{ticker} Correlation Heatmap")
    st.pyplot(fig)

    # Risk analysis
    st.subheader("Risk Analysis")
    returns = data['Daily Return'].dropna()
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    st.write(f"Value at Risk (95% confidence): {var_95:.2%}")
    st.write(f"Conditional Value at Risk (95% confidence): {cvar_95:.2%}")

    # Volume analysis
    st.subheader("Volume Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data.index, data['Volume'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.set_title(f"{ticker} Trading Volume")
    st.pyplot(fig)
# ... (keep all the previous imports and the stock analysis page)

elif page == "Options Analysis":
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

    # Explanation of metrics
    with st.expander("Explanation of Metrics"):
        st.write("""
        - **Option Price**: The current value of the option.
        - **Delta**: The rate of change of the option price with respect to the underlying asset's price.
        - **Gamma**: The rate of change of delta with respect to the underlying asset's price.
        - **Theta**: The rate of change of the option price with respect to time (time decay).
        - **Vega**: The rate of change of the option price with respect to volatility.
        """)

    # Advanced Visualizations
    st.subheader("Advanced Option Analysis Plots")

    # 1. Option Price vs. Stock Price for different times to expiration
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


# Greek vs. Stock Price
    st.write("### Greeks vs. Stock Price")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()  # Flatten the 2x2 array to a 1D array
    
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
    
    plt.tight_layout()
    st.pyplot(fig)

    # 3. Option Price vs. Volatility for different stock prices
    st.write("### Option Price vs. Volatility for Different Stock Prices")
    volatilities = np.linspace(0.1, 1, 100)
    stock_prices = [K * 0.8, K, K * 1.2]  # OTM, ATM, ITM
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in stock_prices:
        option_prices = [calculate_option_metrics(s, K, T, r, vol, option_type)[0] for vol in volatilities]
        ax.plot(volatilities, option_prices, label=f'S = ${s}')
    
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Option Price")
    ax.set_title(f"{option_type.capitalize()} Option Price vs. Volatility")
    ax.legend()
    st.pyplot(fig)

    # 4. Probability of Profit
    st.write("### Probability of Profit")
    def prob_profit(S, K, T, r, sigma, option_type):
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return norm.cdf(d2)
        else:
            return norm.cdf(-d2)

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
elif page == "Portfolio Analysis":
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

    def efficient_frontier(returns, num_portfolios=1000):
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
elif page == "Machine Learning Playground":
    st.title("Machine Learning Playground")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    # Dataset selection
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

    # Model selection
    classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM", "Decision Tree", "Random Forest", "KNN"))

    # Load dataset
    def get_dataset(dataset_name):
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        else:
            data = datasets.load_wine()
        X = data.data
        y = data.target
        return X, y, data.feature_names

    X, y, feature_names = get_dataset(dataset_name)

    # Display dataset info
    st.write(f"## {dataset_name} Dataset")
    st.write("Shape of dataset:", X.shape)
    st.write("Number of classes:", len(np.unique(y)))

    # Parameter selection based on classifier
    def add_parameter_ui(clf_name):
        params = {}
        if clf_name == "SVM":
            C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ("rbf", "linear"))
            params["C"] = C
            params["kernel"] = kernel
        elif clf_name == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 10)
            params["max_depth"] = max_depth
        elif clf_name == "Random Forest":
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            max_depth = st.sidebar.slider("max_depth", 1, 10)
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
        else:
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        return params

    params = add_parameter_ui(classifier_name)

    # Classification
    def get_classifier(clf_name, params):
        if clf_name == "SVM":
            clf = SVC(C=params["C"], kernel=params["kernel"])
        elif clf_name == "Decision Tree":
            clf = DecisionTreeClassifier(max_depth=params["max_depth"])
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
        else:
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        return clf

    clf = get_classifier(classifier_name, params)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and predict
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Classifier: {classifier_name}")
    st.write(f"Accuracy: {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=np.unique(y), y=np.unique(y),
                       title="Confusion Matrix", color_continuous_scale="Viridis")
    st.plotly_chart(fig_cm)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write("Classification Report:")
    st.dataframe(df_report)

    # PCA Visualization
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig_pca = px.scatter(x=x1, y=x2, color=y,
                         labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                         title="PCA Visualization")
    st.plotly_chart(fig_pca)

    # Feature Importance (for tree-based models)
    if classifier_name in ["Decision Tree", "Random Forest"]:
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig_imp = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h'
        ))
        fig_imp.update_layout(title="Feature Importances", xaxis_title="Importance", yaxis_title="Features")
        st.plotly_chart(fig_imp)

        