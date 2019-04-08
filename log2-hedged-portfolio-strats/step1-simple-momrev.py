"""
We try to build a portfolio consisting the eight stocks with different weights,
and hedge the portfolio with the S&P500 to track on alphas.
"""

import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats
from get_daily_pnl import get_daily_pnl_curve

# Load Data

path = os.path.join(os.getcwd(),'..')

df_stocks = pd.read_csv(os.path.join(path, 'data-reader/data/sample-stocks.csv'))
_d1 = df_stocks.set_index(['Date', 'Symbol'])
_d2 = _d1.groupby(level='Symbol').pct_change()*100
_d3 = _d2.rename(lambda name: name+'_pct', axis=1)

df_stocks2 = pd.concat([_d1,_d3], axis=1)
df_stocks2[['Date','Symbol']] = df_stocks2.index.to_frame()

df_ff5 = pd.read_csv(os.path.join(path, 'data-reader/data/sample-ff5.csv'))

df = pd.merge_ordered(df_stocks2, df_ff5, on='Date')
df.Date = pd.to_datetime(df.Date)
df = df.dropna()
df['Ri-Rf'] = df['Adj Close_pct'] - df['RF']
df['Rm-Rf'] = df['Mkt-RF']

# TEST of pnl analysis

# A simple Momentem and Reversal Strategy

df_shares = df[['Date','Symbol']]

weights = df['Adj Close_pct'].apply(
        lambda x: -1/8 if x > 3
                       else (1/8 if x >= 0
                                 else (-1/8 if x < 0 and x > -3
                                            else 1/8)))
df_shares['Weights'] = weights
df_shares = df_shares[df_shares['Symbol'] != '^GSPC']

df_pnl_curve_daily = get_daily_pnl_curve(df_shares = df_shares,
                                         df_returns = df)
plt.figure()
plt.title('A 8-stock long-short momentem & reversal example strategy')
plt.plot(df_pnl_curve_daily)
plt.show()

# calculate returns for FF5 analysis

df_pnl = pd.DataFrame(df_pnl_curve_daily, columns=['pnl_curve'])
df_pnl['Date'] = df_pnl.index

df_ff5['Date'] = pd.to_datetime(df_ff5['Date'])
df_decomp = pd.merge_ordered(df_pnl, df_ff5, on='Date')
df_decomp['ret'] = df_decomp['pnl_curve'].diff(1)
df_decomp['ret'] = df_decomp['ret'].fillna(0)
df_decomp['ret'] *= 100

df_decomp = df_decomp.dropna()
df_decomp['Ri-Rf'] = df_decomp['ret'] - df_decomp['RF']
df_decomp['Rm-Rf'] = df_decomp['Mkt-RF']

# alpha and betas

df_g = df_decomp
params = ['SMB','HML','RMW','CMA']

y = df_g['Ri-Rf']
X = sm.add_constant(df_g[['Rm-Rf']+params])
model = sm.OLS(y, X)
result = model.fit()
result.summary()
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Ri-Rf   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                  0.000
Method:                 Least Squares   F-statistic:                     1.374
Date:                Mon, 08 Apr 2019   Prob (F-statistic):              0.231
Time:                        14:25:12   Log-Likelihood:                -7016.6
No. Observations:                5283   AIC:                         1.405e+04
Df Residuals:                    5277   BIC:                         1.408e+04
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0058      0.013      0.457      0.648      -0.019       0.030
Rm-Rf         -0.0078      0.012     -0.648      0.517      -0.031       0.016
SMB           -0.0182      0.022     -0.829      0.407      -0.061       0.025
HML           -0.0308      0.022     -1.393      0.164      -0.074       0.013
RMW            0.0043      0.029      0.146      0.884      -0.053       0.061
CMA           -0.0324      0.035     -0.919      0.358      -0.102       0.037
==============================================================================
Omnibus:                      925.272   Durbin-Watson:                   1.890
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10224.168
Skew:                           0.500   Prob(JB):                         0.00
Kurtosis:                       9.741   Cond. No.                         3.81
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Interpretation:
    This method is a pure behavioural factor.
    It has no significant market and other risk factor beta,
    nor does it have significant abnormal returns (alpha)
"""

# A parameter mining process (it is not a good idea to mine like this)

df_shares = df[['Date','Symbol']]

thresholds = (0, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 1000)

number_of_colors = len(thresholds)
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              for i in range(number_of_colors)]

plt.figure()
plt.title('Mining 8-stock long-short momentem & reversal strategy with a splitted time')

for i,threshold in enumerate(thresholds):
    weights = df['Adj Close_pct'].apply(
            lambda x: -1/8 if x > threshold
                           else (1/8 if x >= 0
                                     else (-1/8 if x < 0 and x > -threshold
                                                else 1/8)))
    df_shares['Weights'] = weights
    df_shares1 = df_shares.iloc[:int(df_shares.shape[0]/2)][df_shares['Symbol'] != '^GSPC']
    
    df_pnl_curve_daily = get_daily_pnl_curve(df_shares = df_shares1,
                                             df_returns = df)
    plt.plot(df_pnl_curve_daily, label=threshold, ls='-', color=colors[i])

for i,threshold in enumerate(thresholds):
    weights = df['Adj Close_pct'].apply(
            lambda x: -1/8 if x > threshold
                           else (1/8 if x >= 0
                                     else (-1/8 if x < 0 and x > -threshold
                                                else 1/8)))
    df_shares['Weights'] = weights
    df_shares2 = df_shares.iloc[int(df_shares.shape[0]/2):][df_shares['Symbol'] != '^GSPC']
    
    df_pnl_curve_daily = get_daily_pnl_curve(df_shares = df_shares2,
                                             df_returns = df)
    plt.plot(df_pnl_curve_daily, label=None, ls='-.', color=colors[i])

plt.legend()
plt.show()