import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

def safe_get(dict_obj, key, default=None):
    """Safely get a value from a dictionary, returning a default if the key doesn't exist."""
    return dict_obj.get(key, default)

def run():
    st.title("Advanced Stock Analysis")

    # Sidebar for user input
    st.sidebar.header("User Input")
    symbol = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365*4))

    if st.sidebar.button("Analyze"):
        try:
            # Fetch data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            info = stock.info
            
            # Fetch financial data for all available years
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Display company info
            st.header(f"{safe_get(info, 'longName', symbol)} ({symbol})")
            st.subheader(safe_get(info, 'industry', 'N/A'))

            col1, col2, col3 = st.columns(3)
            with col1:
                current_price = safe_get(info, 'currentPrice', data['Close'].iloc[-1] if not data.empty else 'N/A')
                price_change = safe_get(info, 'regularMarketChangePercent', 
                                        ((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100) if not data.empty else 'N/A')
                if isinstance(current_price, (int, float)) and isinstance(price_change, (int, float)):
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
                else:
                    st.metric("Current Price", "N/A", "N/A")
            with col2:
                market_cap = safe_get(info, 'marketCap', 'N/A')
                st.metric("Market Cap", f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else "N/A")
            with col3:
                low = safe_get(info, 'fiftyTwoWeekLow', 'N/A')
                high = safe_get(info, 'fiftyTwoWeekHigh', 'N/A')
                if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    st.metric("52 Week Range", f"${low:.2f} - ${high:.2f}")
                else:
                    st.metric("52 Week Range", "N/A")

            # Financial Metrics
            st.subheader("Key Financial Metrics")
            metrics = pd.DataFrame({
                'Revenue': financials.loc['Total Revenue'],
                'Gross Profit': financials.loc['Gross Profit'],
                'Operating Income': financials.loc['Operating Income'],
                'Net Income': financials.loc['Net Income'],
                'EBITDA': financials.loc['EBITDA'],
                'Free Cash Flow': cash_flow.loc['Free Cash Flow'],
                'Total Assets': balance_sheet.loc['Total Assets'],
                'Total Liabilities': balance_sheet.loc['Total Liabilities Net Minority Interest'],
            })

            # Calculate additional metrics
            metrics['Gross Margin'] = metrics['Gross Profit'] / metrics['Revenue']
            metrics['Operating Margin'] = metrics['Operating Income'] / metrics['Revenue']
            metrics['Net Profit Margin'] = metrics['Net Income'] / metrics['Revenue']
            metrics['EBITDA Margin'] = metrics['EBITDA'] / metrics['Revenue']
            metrics['FCF Margin'] = metrics['Free Cash Flow'] / metrics['Revenue']
            metrics['Debt to Asset Ratio'] = metrics['Total Liabilities'] / metrics['Total Assets']

            # Display metrics
            st.dataframe(metrics)

            # Create a comprehensive profit situation plot
            fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=("Revenue and Profits", "Profit Margins", 
                                                "Free Cash Flow", "Debt to Asset Ratio"))

            # Revenue and Profits
            fig.add_trace(go.Bar(x=metrics.index, y=metrics['Revenue'], name='Revenue'), row=1, col=1)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Gross Profit'], name='Gross Profit', mode='lines+markers'), row=1, col=1)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Operating Income'], name='Operating Income', mode='lines+markers'), row=1, col=1)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Net Income'], name='Net Income', mode='lines+markers'), row=1, col=1)

            # Profit Margins
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Gross Margin'], name='Gross Margin', mode='lines+markers'), row=1, col=2)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Operating Margin'], name='Operating Margin', mode='lines+markers'), row=1, col=2)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Net Profit Margin'], name='Net Profit Margin', mode='lines+markers'), row=1, col=2)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['EBITDA Margin'], name='EBITDA Margin', mode='lines+markers'), row=1, col=2)

            # Free Cash Flow
            fig.add_trace(go.Bar(x=metrics.index, y=metrics['Free Cash Flow'], name='Free Cash Flow'), row=2, col=1)
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['FCF Margin'], name='FCF Margin', mode='lines+markers', yaxis='y2'), row=2, col=1)

            # Debt to Asset Ratio
            fig.add_trace(go.Scatter(x=metrics.index, y=metrics['Debt to Asset Ratio'], name='Debt to Asset Ratio', mode='lines+markers'), row=2, col=2)

            # Update layout
            fig.update_layout(height=800, title_text="Comprehensive Financial Analysis")
            fig.update_yaxes(title_text="USD", row=1, col=1)
            fig.update_yaxes(title_text="Ratio", row=1, col=2)
            fig.update_yaxes(title_text="USD", row=2, col=1)
            fig.update_yaxes(title_text="Ratio", secondary_y=True, row=2, col=1)
            fig.update_yaxes(title_text="Ratio", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

            # Stock price chart
            if not data.empty:
                st.subheader("Stock Price Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

            # Financial Ratios
            st.subheader("Financial Ratios")
            ratios = pd.DataFrame({
                'P/E Ratio': [info.get('trailingPE', 'N/A')],
                'Forward P/E': [info.get('forwardPE', 'N/A')],
                'PEG Ratio': [info.get('pegRatio', 'N/A')],
                'Price to Book Ratio': [info.get('priceToBook', 'N/A')],
                'Price to Sales Ratio': [info.get('priceToSalesTrailing12Months', 'N/A')],
                'Dividend Yield': [info.get('dividendYield', 'N/A')],
            })
            st.dataframe(ratios)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check if the stock symbol is valid and try again.")

if __name__ == "__main__":
    run()