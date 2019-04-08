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

def build_strat(make_share, make_pnl, stop_one=False):
    for g,df_g in df_fits.groupby('symbol'):
        print(g)
        df_g.index = pd.to_datetime(df_g['date'])
        
        make_share(df_g)
        make_pnl(g,df_g)
        
        if stop_one:
            global df_g_obs
            df_g_obs = df_g
            return

# Load Data

df_fits = pd.read_csv('data/sample-ff5-decomp-stocks.csv')

# Model I: Random Walk model as Benchmark
def make_share1(df_g):
    """
    Basic RW Model
    """
    df_g['share'] = df_g['indiv_part'].apply(\
        lambda num:1 if num>0 else (-1 if num<0 else 0))

build_strat(make_share1, make_pnl, stop_one=False)

# Model II: Random Walk model with some threshold
def make_share2(df_g):
    """
    Reverse strat with a threshold
    """
    df_g['share'] = df_g['indiv_part'].apply(\
        lambda num:-1 if num>2 else (1 if num<-2 else 0))

build_strat(make_share2, make_pnl, stop_one=False)
