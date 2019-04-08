import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

def make_pnl(g,df_g):
    shares = df_g['share'].iloc[:df_g.shape[0]-1]
    rets = df_g['indiv_part'].iloc[1:]
    shares.index = rets.index

    df_g['pnl'] = [0] + (shares*rets).tolist()    
    df_g['pnl_curve'] = (1+df_g['pnl']/100).cumprod()
    
    plt.figure()
    plt.title('P&L Curve of RW model for abnormal return analysis for %s'%g)
    plt.plot(df_g['pnl_curve'])
    plt.show()    

def build_strat(df, make_share, make_pnl, stop_one=False):
    for g,df_g in df.groupby('symbol'):
        print(g)
        df_g.index = pd.to_datetime(df_g['date'])
        
        make_share(df_g)
        make_pnl(g,df_g)
        
        if stop_one:
            global df_g_obs
            df_g_obs = df_g
            return

# Load Data
path = os.path.join(os.getcwd(),'..')

df_fits = pd.read_csv('data/sample-ff5-decomp-stocks.csv')
df_stocks = pd.read_csv(os.path.join(path, 'data-reader/data/sample-stocks.csv'))

df = pd.merge_ordered(df_fits, df_stocks,
                      left_on=['date','symbol'],
                      right_on=['Date','Symbol'],
                      how='left')
df_merged = df[df_fits.columns.tolist()+['Volume']]

# Model III: Volume Factor model
def make_share1(df_g):
    """
    Volume Factor Model
    """
    df_g['share'] = df_g['indiv_part'].apply(\
        lambda num:1 if num>0 else (-1 if num<0 else 0))

build_strat(df_merged, make_share1, make_pnl, stop_one=True)

# Low R2, probably not a good model;
# Try to check their relationships

df_g = df_g_obs
y = df_g['indiv_part']
X = sm.add_constant(np.log(df_g['Volume']+1))
model = sm.GLS(y,X)
result = model.fit()
print(result.summary())

plt.figure()
plt.title('Relationship between Volume and Abnormal Return of a security')
plt.scatter(np.log(df_g['Volume']+1), y)
plt.xlabel('log-volume')
plt.ylabel('abnormal-returns')
plt.ylim(-10, 10)
plt.show()

# So far, we know high volume indicates high abnormal return volatility!


