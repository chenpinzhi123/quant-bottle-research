import os
import pandas as pd

path = os.path.join(os.getcwd(),'..')
df = pd.read_csv(os.path.join(path, 'data-reader/data/sample-stocks.csv'),
                 header=[0,1])

