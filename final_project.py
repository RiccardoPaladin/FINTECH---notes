#pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import date
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import Lasso
import seaborn as sn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import statsmodels.api as sm


st.title('Stock fundamental analysis')
st.text('In this web app you can insert stock tickers and obtain several results...')

st.text('Insert a series of tickers with comma')

tickers_input = st.text_input('Enter here the tickers','')
start_date = st.text_input('Enter here the start date','')
end_date = st.text_input('Enter here the end date','')

Data = data.DataReader(tickers_input, 'yahoo', start_date, end_date)
Stocks_prices = Data['Adj Close']
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
Stocks_prices = Stocks_prices.reindex(all_weekdays)
Stocks_prices = Stocks_prices.fillna(method='ffill')
prices = Stocks_prices.head()

st.markdown(
    f"""
    {prices}
    """
)

Stocks = Stocks_prices.pct_change()
Returns = Stocks.dropna()

st.markdown(
    f"""
    {Returns}
    """
)