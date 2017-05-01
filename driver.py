import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet

playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                          '2010-01-24', '2010-02-07', '2011-01-08',
                          '2013-01-12', '2014-01-12', '2014-01-19',
                          '2014-02-02', '2015-01-11', '2016-01-17',
                          '2016-01-24', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1,
})
superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1,
})

newYears = pd.DataFrame({
    'holiday': 'New Years',
    'ds': pd.to_datetime(
        ['1990-01-01', '1991-01-01', '1992-01-01', '1993-01-01', '1994-01-01', '1995-01-01', '1996-01-01', '1997-01-01',
         '1998-01-01', '1999-01-01', '2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01',
         '2005-01-01']),
    'lower_window': 0,
    'upper_window': 1,
})
memorialDay = pd.DataFrame({
    'holiday': 'Memorial Day',
    'ds': pd.to_datetime(
        ['1990-05-28', '1991-05-27', '1992-05-25', '1993-05-31', '1994-05-30', '1995-05-29', '1996-05-27', '1997-05-26',
         '1998-05-25', '1999-05-31', '2000-05-29', '2001-05-28', '2002-05-27', '2003-05-26', '2004-05-31',
         '2005-05-30']),
    'lower_window': 0,
    'upper_window': 1,
})
independenceDay = pd.DataFrame({
    'holiday': 'Independence Day',
    'ds': pd.to_datetime(
        ['1990-07-04', '1991-07-04', '1992-07-04', '1993-07-04', '1994-07-04', '1995-07-04', '1996-07-04', '1997-07-04',
         '1998-07-04', '1999-07-04', '2000-07-04', '2001-07-04', '2002-07-04', '2003-07-04', '2004-07-04',
         '2005-07-04', ]),
    'lower_window': 0,
    'upper_window': 0,
})
christmas = pd.DataFrame({
    'holiday': 'Christmas',
    'ds': pd.to_datetime(
        ['1990-12-24', '1991-12-24', '1992-12-24', '1993-12-24', '1994-12-24', '1995-12-24', '1996-12-24', '1997-12-24',
         '1998-12-24', '1999-12-24', '2000-12-24', '2001-12-24', '2002-12-24', '2003-12-24', '2004-12-24',
         '2005-12-24']),
    'lower_window': -1,
    'upper_window': 1,
})

holidays = pd.concat((newYears, memorialDay, independenceDay, christmas))

df = pd.read_csv('data/stock/SP500-95-12.csv')


def find_the_future_nh(data_frame, period):
    nh_m = Prophet()
    nh_m.fit(data_frame)
    nh_future = nh_m.make_future_dataframe(periods=period)
    nh_forcast = nh_m.predict(nh_future)
    return nh_forcast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

out = ""

out = out + find_the_future_nh(df, 30)
out = out + find_the_future_nh(df, 180)
out = out + find_the_future_nh(df, 360)
out = out + find_the_future_nh(df, 540)
out = out + find_the_future_nh(df, 720)
out = out + find_the_future_nh(df, 900)
out = out + find_the_future_nh(df, 1080)

print(out)

# m = Prophet(holidays=holidays)
# m.fit(df)
# future = m.make_future_dataframe(periods=1096)
# print(future.tail())
# forecast = m.predict(future)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# m.plot(forecast)
# for cp in m.changepoints:
#     plt.axvline(cp, c='gray', ls='--', lw=2)
# plt.show()
