import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv('data/example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
print(df.head())
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
print(future.tail())
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
m.plot(forecast)
plt.show()
