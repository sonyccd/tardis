import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('data/example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
print(df.head())
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
print(future.tail())
