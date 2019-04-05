from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

# Initialization
tickers = ['AAPL', 'MSFT', '^GSPC']
start_date = '2014-01-01'
end_date = '2018-12-31'

# Fetch Data from yahoo api
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

# Show the head of panel data
print(panel_data.head(10))
