from setuptools import setup, find_packages

setup(
    name="quant_analysis_app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "yfinance",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "statsmodels",
        "scipy",
        "plotly",
    ],
    entry_points={
        "console_scripts": [
            "quant_analysis_app=main:main",
        ],
    },
)