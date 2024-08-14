import streamlit as st
from app import stock_analysis, options_analysis, portfolio_analysis, ml_playground

st.set_page_config(page_title="Advanced Quant Analysis App", layout="wide")

# CSS for the logo links
logo_css = """
<style>
.logo-link {
    display: inline-block;
    margin-right: 10px;
}

.logo-img {
    width: 30px;
    height: 30px;
    border-radius: 5px;
}
</style>
"""

# HTML for the logo links
logo_html = f"""
<div>
    <a href="www.linkedin.com/in/alexander-johae-a82363211" target="_blank" class="logo-link">
        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn" class="logo-img">
    </a>
    <a href="https://github.com/alex8098" target="_blank" class="logo-link">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" class="logo-img">
    </a>
</div>
"""

# Render the CSS and HTML
st.sidebar.markdown(logo_css, unsafe_allow_html=True)
st.sidebar.markdown(logo_html, unsafe_allow_html=True)

# Add a divider
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Choose a page", ["Stock Analysis", "Options Analysis", "Portfolio Analysis", "Machine Learning Playground"])

if page == "Stock Analysis":
    stock_analysis.run()
elif page == "Options Analysis":
    options_analysis.run()
elif page == "Portfolio Analysis":
    portfolio_analysis.run()
elif page == "Machine Learning Playground":
    ml_playground.run()

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 Your Name. All rights reserved.")