from pandas_datareader import data

# Initialization
tickers = ['^GSPC', 'XOM', 'AAPL', 'INTC', 'GS', 'GE', 'AXP', 'WMT', 'PG']
start_date = '1998-01-01'
end_date = '2018-12-31'

# Fetch Data from yahoo api
panel_data = data.DataReader(tickers, 'google', start_date, end_date)

# Show the head of panel data
print(panel_data.head(10))

# Store the sample data
#panel_data.to_csv('data/sample-stocks.csv')

"""
Note:
    Data of GS is not available from 1998-01-01 to 1999-05-05
"""