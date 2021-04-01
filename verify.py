import pandas as pd
import csv
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, pacf

dfss = pd.read_excel('Dataset.xlsx', sheet_name='SS(Ave)')
dfss['Datetime'] = pd.to_datetime(dfss['Date'])
dfss = dfss.set_index('Datetime')
dfss = dfss.drop(['YYYY-MM','Date'], axis=1)

dffe = pd.read_excel('Dataset.xlsx', sheet_name='FE')
dffe['Datetime'] = pd.to_datetime(dffe['Date'])
dffe = dffe.set_index('Datetime')
dffe = dffe.drop(['YYYY-MM','Date'], axis=1)

dfat = pd.read_excel('Dataset.xlsx', sheet_name='AT(Ave)')
dfat['Datetime'] = pd.to_datetime(dfat['Date'])
dfat = dfat.set_index('Datetime')
dfat = dfat.drop(['YYYY-MM','Date'], axis=1)

dfss_tn = dfss[['BOD', 'NH3-N', 'TN','PH']]
dfat_tn=dfat[['MLSS','AT_Temp']]
dffe_tn = dffe[['TN']]

dffe_tn.columns = ['OUTPUT TN']

dfss_tnMA = dfss_tn.rolling(5, min_periods=1).mean()
dfat_tnMA=dfat_tn.rolling(5, min_periods=1).mean()
dffe_tnMA=dffe_tn.rolling(5, min_periods=1).mean()

dfss_tn.fillna(dfss_tnMA,inplace=True)
dfat_tn.fillna(dfat_tnMA,inplace=True)
dffe_tn.fillna(dffe_tnMA,inplace=True)

tn_data = pd.concat([dfss_tn,dfat_tn,dffe_tn], axis=1)
tn_data = tn_data.dropna()

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
test_stationarity(tn_data['OUTPUT TN'])

def difference(dataset, interval=1):
    index = list(dataset.index)
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset["OUTPUT TN"][i] - dataset["OUTPUT TN"][i - interval]
        diff.append(value)
    return (diff)

diff = difference(tn_data)
plt.plot(diff)
plt.show()

tn_log = np.log(tn_data["OUTPUT TN"])
plt.title('Log of the data')
plt.plot(tn_log)
plt.show()

moving_avg = tn_log.rolling(30).mean()
plt.plot(tn_log)
plt.title('Moving average')
plt.plot(moving_avg, color='red')
plt.show()

tn_log_moving_avg_diff = tn_log - moving_avg
tn_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(tn_log_moving_avg_diff)

expwighted_avg = tn_log.ewm(halflife=12).mean()
#parameter halflife is used to define the amount of exponential decay
plt.plot(tn_log)
plt.plot(expwighted_avg, color='red')
plt.show()

tn_log_ewma_diff = tn_log - expwighted_avg
test_stationarity(tn_log_ewma_diff)

tn_log_diff = tn_log - tn_log.shift()
plt.plot(tn_log_diff)
plt.show()

tn_log_diff.dropna(inplace=True)
test_stationarity(tn_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(tn_data['OUTPUT TN'], model='additive',period=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

ax1 = plt.subplot(411)
ax1.plot(tn_data['OUTPUT TN'], label='Original')
plt.legend(loc='best')
ax1.set_ylim([3, 14])

ax2 = plt.subplot(412)
ax2.plot(trend, label='Trend')
plt.legend(loc='best')
ax2.set_ylim([3, 14])

ax3 = plt.subplot(413)
ax3.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
ax3.set_ylim([-3, 5])

ax4 = plt.subplot(414)
ax4.plot(residual, label='Residuals')
plt.legend(loc='best')
ax4.set_ylim([-3, 5])
plt.show()
plt.tight_layout