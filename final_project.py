#pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import date
import yfinance as yf
import seaborn as sn

st.set_page_config(
    page_title="Stock fundamental analysis")

st.title('Stock fundamental analysis')
st.text('In this web app you can insert stock tickers and obtain several results...')

st.text('Insert a series of tickers with comma')

#tickers_input =list(st.text_input('Enter here the tickers',''))
#start_date = st.text_input('Enter here the start date','')
#end_date = st.text_input('Enter here the end date','')
start_date = '01-01-2020'
end_date = '03-28-2022'
tickers_input = ['SPY', 'AAPL']
Data = data.DataReader(tickers_input, 'yahoo', start_date, end_date)
Stocks_prices = Data['Adj Close']
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
Stocks_prices = Stocks_prices.reindex(all_weekdays)
Stocks_prices = Stocks_prices.fillna(method='ffill')
Stocks_price = st.dataframe(Stocks_prices)

st.markdown(
    f"""
    {Stocks_price}
    """
)

#Stocks_prices = pd.DataFrame(Stocks_prices)
corrMatrix = Stocks_prices.corr()
plt.figure(figsize=(15,10))
plot = st.pyplot(sn.heatmap(corrMatrix, annot=False))


st.markdown(
    f"""
    {plot}
    """
)