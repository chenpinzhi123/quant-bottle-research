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
df = df.dropna()

# CAPM-based analysis

df['Ri-Rf'] = df['Adj Close_pct'] - df['RF']
df['Rm-Rf'] = df['Mkt-RF']

show_sets = []
for g,df_g in df.groupby('Symbol'):
    y = df_g['Ri-Rf']
    X = sm.add_constant(df_g['Rm-Rf'])
    model = sm.OLS(y, X)
    result = model.fit()
    
    show_set = {'Symbol':g,
                'alpha':result.params[0],
                'alpha_std':result.HC0_se[0],
                'beta':result.params[1],
                'beta_std':result.HC0_se[1],
                'R2':result.rsquared}
    show_sets.append(show_set)
results_capm = pd.DataFrame(show_sets,
                            columns=['Symbol','alpha','alpha_std',
                                     'beta','beta_std','R2'])

# Visualization of CAPM model
_r = results_capm[results_capm['Symbol']=='AAPL']
xarray = np.linspace(-15, 15)
yarray = _r.alpha[0] + _r.beta[0]*xarray
plt.figure()
plt.title('CAPM decomposition of AAPL from 1998 to 2018')
plt.scatter(df[df['Symbol']=='AAPL']['Rm-Rf'],
            df[df['Symbol']=='AAPL']['Ri-Rf'])
plt.plot(xarray, yarray, color='red', label='L2 Reg')
plt.xlabel('Rm-Rf')
plt.ylabel('Ri-Rf')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.legend()
plt.show()

# TODO: use LAD regression via specified ML models

# FF5-based analysis

params = ['SMB','HML','RMW','CMA']
show_sets = []
for g,df_g in df.groupby('Symbol'):
    y = df_g['Ri-Rf']
    X = sm.add_constant(df_g[['Rm-Rf']+params])
    model = sm.OLS(y, X)
    result = model.fit()
    
    show_set = {'Symbol':g,
                'alpha':result.params[0],
                'alpha_std':result.HC0_se[0],
                'beta':result.params[1],
                'beta_std':result.HC0_se[1],
                'R2':result.rsquared}
    for param in params:
        show_set[param] = result.params[param]
        show_set['%s_std'%param] = result.HC0_se[param]
    show_sets.append(show_set)

results_ff5 = pd.DataFrame(show_sets,
                           columns=['Symbol','alpha','alpha_std',
                                    'beta','beta_std',
                                    'SMB','SMB_std',
                                    'HML','HML_std',
                                    'RMW','RMW_std',
                                    'CMA','CMA_std','R2'])
