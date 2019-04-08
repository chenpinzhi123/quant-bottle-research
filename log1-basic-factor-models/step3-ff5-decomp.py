import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

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

# Decomposition of market risk from individual returns
# Because time-shift of beta is frequent, we roll-estimate betas each day
# from data within a year.

params = ['SMB','HML','RMW','CMA']
window = 252
fits = []
for g,df_g in df.groupby('Symbol'):
    print(g)
    df_g.index = pd.to_datetime(df_g['Date'])
    if df_g.shape[0] < window:
        continue
    for i in range(df_g.shape[0]-window):
        df_gw = df_g.iloc[i:i+window]
        y = df_gw['Ri-Rf']
        X = sm.add_constant(df_gw[['Rm-Rf']+params])
        model = sm.OLS(y, X)
        result = model.fit()
        # Model the next-day individual returns
        mkt_part = result.params[0]+sum(result.params[1:]*df_g[['Rm-Rf']+params].iloc[i+window])
        real_part = df_g['Ri-Rf'].iloc[i+window]
        indiv_part = real_part - mkt_part
        date = df_g['Date'].iloc[i+window]
        symbol = g
        fits.append([date,symbol,mkt_part,real_part,indiv_part])

df_fits = pd.DataFrame(fits, columns=['date','symbol','mkt_part','real_part','indiv_part'])

# Graphical illustration of ff5 risk decomposition

y = df_fits['real_part']
X = sm.add_constant(df_fits['mkt_part'])
model = sm.OLS(y, X)
result = model.fit()

xarray = np.linspace(-20, 20)
yarray = xarray
yarray_real = result.params[0]+result.params[1]*xarray

plt.figure()
plt.title('market decomposed returns and real returns')
plt.xlabel('market decomposed returns')
plt.ylabel('real returns')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.scatter(df_fits['mkt_part'], df_fits['real_part'], label=None)
plt.plot(xarray, yarray, label='in theory', color='red', ls='-.')
plt.plot(xarray, yarray_real, label='in reality', color='green', ls=':')
plt.legend()
plt.show()

t_stat = (result.params[1]-1)/result.bse[1]

# Graphically, ff5 is a good method of market risk decomposition;
# though statistically it is not (but close to and good enough)

df_fits.to_csv('data/sample-ff5-decomp-stocks.csv', index=False)

