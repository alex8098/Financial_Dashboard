import yfinance as yf
import pandas as pd

def load_data(tickers, start_date, end_date):
    if isinstance(tickers, str):
        tickers = [tickers]
    
    data = yf.download(tickers, start=start_date, end=end_date)
    
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([data.columns, tickers])
    
    return data

def calculate_returns(data):
    return data['Adj Close'].pct_change().dropna()