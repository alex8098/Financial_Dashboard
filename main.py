import streamlit as st
from app import stock_analysis, options_analysis, portfolio_analysis, ml_playground

st.set_page_config(page_title="Advanced Quant Analysis App", layout="wide")

page = st.sidebar.selectbox("Choose a page", ["Stock Analysis", "Options Analysis", "Portfolio Analysis", "Machine Learning Playground"])

if page == "Stock Analysis":
    stock_analysis.run()
elif page == "Options Analysis":
    options_analysis.run()
elif page == "Portfolio Analysis":
    portfolio_analysis.run()
elif page == "Machine Learning Playground":
    ml_playground.run()