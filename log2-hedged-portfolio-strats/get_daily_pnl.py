"""
We try to build a portfolio consisting the eight stocks with different weights,
and hedge the portfolio with the S&P500 to track on alphas.
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats

def get_daily_pnl_curve(df_shares, df_returns):
    """
    dataframe must contain columns with exactly the same name:
        df_shares:  Date, Symbol and Weights
        df_returns: Date, Symbol and Adj Close_pct
    """

    df_merged = pd.merge_ordered(df_shares, df_returns,
                                 on=['Date','Symbol'], how='left')
    df_merged = df_merged[['Date','Symbol','Weights','Adj Close_pct']]
    
    grouper = df_merged.groupby('Symbol')
    def calculate_pnl(df_g): #,fill_value=0)
        df_helper = 1 + df_g['Weights'].shift(1)*df_g['Adj Close_pct']/100
        #df_helper.iloc[0] = df_g['Weights'].iloc[0]
        df_helper.fillna(1, inplace=True)
        df_helper.name = 'pnl'
        return pd.concat([df_g, df_helper], axis=1)
    df_merged = grouper.apply(calculate_pnl)
    
    pnl_curve = df_merged.groupby('Symbol')['pnl'].cumprod()
    pnl_curve.name = 'pnl_curve'
    df_pnl_curve = pd.concat([df_merged,pnl_curve],axis=1)
    
    df_pnl_curve_daily = df_pnl_curve.groupby('Date').apply(lambda df_g:1+(df_g['pnl_curve']-1).sum())
    return df_pnl_curve_daily

if __name__ == '__main__':
    """
    Test a sample.
    """
    
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
    
    df_shares = pd.DataFrame({'Date':['1998-01-02','1998-01-02',
                                      '1998-01-05','1998-01-05',
                                      '1998-01-06','1998-01-06',
                                      '1998-01-07','1998-01-07',],
                              'Symbol':['AAPL','INTC','AAPL','INTC',
                                        'AAPL','INTC','AAPL','INTC'],
                              'Weights':[  0,  0,  2, -1,
                                         0.5,0.5,  0,  0]})
    df_shares['Date'] = pd.to_datetime(df_shares['Date'])
    
    df_pnl_curve_daily = get_daily_pnl_curve(df_shares = df_shares,
                                             df_returns = df)
    plt.plot(df_pnl_curve_daily)
