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


Portfolio_selected = Stocks
p_ret = []
p_vol = []
p_weights = []
num_assets = len(Portfolio_selected.columns)
num_portfolios = 10000
cov_matrix = Portfolio_selected.apply(lambda x: np.log(1 + x)).cov()

mean_returns_annual = []
for (columnName, columnData) in Portfolio_selected.iteritems():
    means_a = columnData.mean() * 252
    mean_returns_annual.append(means_a)

for portfolio in range(num_portfolios):
    weights = np.random.uniform(0.05, 0.15, num_assets)
    weights = weights / np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, mean_returns_annual)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  # Portfolio Variance
    sd = np.sqrt(var)  # Daily standard deviation
    ann_sd = sd * np.sqrt(252)  # Annual standard deviation = volatility
    p_vol.append(ann_sd)

data = {'Returns': p_ret, 'Volatility': p_vol}

for counter, symbol in enumerate(Portfolio_selected.columns.tolist()):
    # print(counter, symbol)
    data[symbol] = [w[counter] for w in p_weights]

portfolios_generated = pd.DataFrame(data)
portfolios_generated.head()

min_vol_port = portfolios_generated.iloc[portfolios_generated['Volatility'].idxmin()]
min_vol_port = st.dataframe(min_vol_port)

st.text('Weights for the minimum variance portfolio ')

st.markdown(
    f"""
    {min_vol_port}
    """
)


optimal_risky_port = portfolios_generated.iloc[((portfolios_generated['Returns'])/
                                                portfolios_generated['Volatility']).idxmax()]

optimal_risky_port = st.dataframe(optimal_risky_port)
st.text('Weights for the maximum Sharpe Ratio  portfolio ')

st.markdown(
    f"""
    {optimal_risky_port}
    """
)


st.text('Predictions ')


st.set_page_config(
    page_title="Stock fundamental analysis")

st.title('Stock fundamental analysis')
st.text('In this web app you can insert stock tickers and obtain several results...')

st.text('Insert a series of tickers with comma')

tickers_input = st.text_input('Enter here the tickers and in the first position the benchmark','').split()

start_date = st.text_input('Enter here the start date','')
end_date = st.text_input('Enter here the end date','')
#start_date = '01-01-2020'
#end_date = '03-28-2022'
#tickers_input = [tickers_input1, tickers_input2]
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

price_chart = plt.plot(Stocks_prices)

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
for i in range(len(Stocks.columns)):
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


