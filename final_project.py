#pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import date
import yfinance as yf
import seaborn as sn
from sklearn.linear_model import LinearRegression

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
tickers_input = ['SPY', 'AAPL', 'TSLA']
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

price_chart = st.line_chart(pd.DataFrame(Stocks_prices['SPY']))

st.markdown(
    f"""
    {price_chart}
    """
)



Stocks = Stocks_prices.pct_change()
Stocks = Stocks.dropna()

fundamentals = []
mean_ret = []
for (columnName, columnData) in Stocks.iteritems():
    means = columnData.mean() * 252
    mean_ret.append(means)
    stds = columnData.std() * (252 ** 0.5)
    sharpe = means / stds
    fundamentals.append(
        {'Stock': columnName,
         'Mean': means,
         'Standard Dev': stds,
         'Sharpe': sharpe

         }
    )

reg_data = []
for i in range(3):
    model = LinearRegression()
    X = Stocks.iloc[0:, 0].to_numpy().reshape(-1, 1)
    Y = Stocks.iloc[0:, i].to_numpy().reshape(-1, 1)
    reg = model.fit(X, Y)
    alpha = float(reg.intercept_)
    beta = float(reg.coef_)
    reg_data.append(
        {'Alpha': alpha,
         'Beta': beta

         })
fundamentals = pd.DataFrame(fundamentals)
reg_data = pd.DataFrame(reg_data)
fundamentals = fundamentals.join(reg_data)
fundamentals = fundamentals.sort_values(by='Sharpe', ascending=False)
fundamental = st.dataframe(fundamentals)
st.markdown(
    f"""
    {fundamental}
    """
)


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.heatmap(Stocks.corr(), ax=ax)
corr = st.write(fig)

st.markdown(
    f"""
    {corr}
    """
)


