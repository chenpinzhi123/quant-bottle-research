import os
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats

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
                'alpha_std':result.bse[0],
                'beta':result.params[1],
                'beta_std':result.bse[1],
                'R2':result.rsquared}
    for param in params:
        show_set[param] = result.params[param]
        show_set['%s_std'%param] = result.bse[param]
    show_sets.append(show_set)

results_ff5 = pd.DataFrame(show_sets,
                           columns=['Symbol','alpha','alpha_std',
                                    'beta','beta_std',
                                    'SMB','SMB_std',
                                    'HML','HML_std',
                                    'RMW','RMW_std',
                                    'CMA','CMA_std','R2'])

# Will market beta in the ff5 change over time? By how much is it changing?
betas = []
for sym,df_sym in df.groupby('Symbol'):
    for g,df_g in df_sym.groupby(df_sym['Date'].dt.year):
        y = df_g['Ri-Rf']
        X = sm.add_constant(df_g[['Rm-Rf']+params])
        model = sm.OLS(y, X)
        result = model.fit()
        betas.append({'Symbol':sym,
                      'year':pd.datetime(g,12,31),
                      'beta':result.params[1]})
results_betas2 = pd.DataFrame(betas)

# In the graph, show AAPL, GE and ^GSPC only
plt.figure()
plt.title('Yearly Market Beta in FF5 for some firms and S&P500(^GSPC)')
for sym,res_sym in results_betas2.groupby('Symbol'):
    if sym in ('AAPL','GE','^GSPC'):
        plt.plot(res_sym['year'], res_sym['beta'], label=sym)
plt.ylabel('market beta')
plt.xlabel('year')
plt.legend()
plt.show()
# -> graphically, market beta in ff5 can also be volatile

# A chow's test for FF5
"""
Chow's test:
    F = [RSS(C)-(RSS(1)+RSS(2))/k]/[(RSS(1)+RSS(2))/(N1+N2-2*k)]
"""
chow_results = []
for g,df_g in df.groupby('Symbol'):
    y = df_g['Ri-Rf']
    X = sm.add_constant(df_g[['Rm-Rf']+params])
    model_C = sm.OLS(y, X)
    result_C = model_C.fit()
    rssc = result_C.ssr # sum of squared residuals
    
    N1 = int(df_g.shape[0]/2)
    y1 = df_g['Ri-Rf'][:N1]
    X1 = sm.add_constant(df_g[['Rm-Rf']+params][:N1])
    model1 = sm.OLS(y1, X1)
    result1 = model1.fit()
    rss1 = result1.ssr # sum of squared residuals

    y2 = df_g['Ri-Rf'][N1:]
    X2 = sm.add_constant(df_g[['Rm-Rf']+params][N1:])
    model2 = sm.OLS(y2, X2)
    result2 = model2.fit()
    rss2 = result2.ssr # sum of squared residuals
    
    k = X.shape[1]
    N2 = X2.shape[0]
    F = ((rssc-rss1-rss2)/k)/((rss1+rss2)/(N1+N2-2*k))
    p_value = 1-stats.f.cdf(F,k,N1+N2-2*k)
    
    chow_results.append({'Symbol':g,
                         'F-stats':F,
                         'p-value':p_value})
results_chow2 = pd.DataFrame(chow_results, columns=['Symbol','F-stats','p-value'])
# -> statistically, market beta in ff5 can also be volatile
