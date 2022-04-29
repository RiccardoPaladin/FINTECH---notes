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
from scipy.stats import kurtosis, skew

st.set_page_config(
    page_title="Stock fundamental analysis")

st.title('Stock fundamental analysis')
st.text('In this web app you can insert stock tickers and obtain several results...')

st.text('Insert a series of tickers with comma')

tickers_input = st.text_input('Enter here the tickers','')
start_date = st.text_input('Enter here the start date','')
end_date = st.text_input('Enter here the end date','')
start_date = '01-01-2020'
end_date = '03-28-2022'
#tickers_input = ['SPY', 'AAPL', 'TSLA']
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

price_chart = st.line_chart(pd.DataFrame(Stocks_prices.iloc[0:,1]))

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

# Performance Metric Functions

def sharpe(returns, risk_free=0):
    adj_returns = returns - risk_free
    return (np.nanmean(adj_returns)) / np.nanstd(adj_returns, ddof=1)

def downside_risk(returns, risk_free=0):
    adj_returns = returns - risk_free
    sqr_downside = np.square(np.clip(adj_returns, np.NINF, 0))
    return np.sqrt(np.nanmean(sqr_downside))

def sortino(returns, risk_free=0):
    adj_returns = returns - risk_free
    drisk = downside_risk(adj_returns)

    if drisk == 0:
        return np.nan

    return (np.nanmean(adj_returns)) / drisk

def Omega(returns,threshold):
  dailyThresh = (threshold + 1) ** np.sqrt(1 / 252) - 1

  returns['Excess'] = returns['Portfolio Returns'] - dailyThresh

  ret_PosSum = (returns[returns['Excess'] > 0].sum())['Excess']
  ret_NegSum = (returns[returns['Excess'] < 0].sum())['Excess']

  omega = ret_PosSum / (-ret_NegSum)

  return omega

def get_kurtosis(returns):
    rets = returns.to_numpy()
    kurt = kurtosis(rets, fisher = True)

    return kurt[0]

def get_skew(returns):
    rets = returns.to_numpy()
    skewness = skew(rets)

    return skewness[0]
