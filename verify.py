import pandas as pd
import numpy as np
import matplotlib.pylab as plt

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

dfss_fin= dfss[['BOD', 'NH3-N', 'TN','PH']]
dfat_fin=dfat[['MLSS','AT_Temp']]
dfss_finMA = dfss_fin.rolling(5, min_periods=1).mean()
dfat_finMA=dfat_fin.rolling(5, min_periods=1).mean()
dfss_fin.fillna(dfss_finMA,inplace=True)
dfat_fin.fillna(dfat_finMA,inplace=True)

# TOTAL NITROGEN
dffe_tn = dffe[['TN']]
dffe_tn.columns = ['OUTPUT TN']
dffe_tnMA=dffe_tn.rolling(5, min_periods=1).mean()
dffe_tn.fillna(dffe_tnMA,inplace=True)

tn_data = pd.concat([dfss_fin,dfat_fin,dffe_tn], axis=1)
tn_data = tn_data.dropna()

# AMMONIA
dffe_nh3 = dffe[['NH3-N']]
dffe_nh3MA=dffe_nh3.rolling(5, min_periods=1).mean()
dffe_nh3.fillna(dffe_nh3MA,inplace=True)
dffe_nh3.columns = ['OUTPUT NH3']
nh3_data = pd.concat([dfss_fin,dfat_fin,dffe_nh3], axis=1)

nh3_data.drop(nh3_data.loc[nh3_data['OUTPUT NH3'] <0.5000001 ].index, inplace=True)
nh3_data = nh3_data.dropna()

# Biological Oxygen Demand
dffe_bod = dffe[['BOD']]
dffe_bod.columns = ['OUTPUT BOD']
dffe_bodMA=dffe_bod.rolling(5, min_periods=1).mean()
dffe_bod.fillna(dffe_bodMA,inplace=True)

bod_data = pd.concat([dfss,dfat,dffe_bod], axis=1)

bod_data.drop(bod_data.loc[bod_data['OUTPUT BOD'] < 5.00000001 ].index, inplace=True)

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

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(bod_data['OUTPUT BOD'], model='additive',period=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

ay1 = plt.subplot(411)
ay1.plot(bod_data['OUTPUT BOD'], label='Original')
plt.legend(loc='best')
ay1.set_ylim([3, 14])

ay2 = plt.subplot(412)
ay2.plot(trend, label='Trend')
plt.legend(loc='best')
ay2.set_ylim([3, 14])

ay3 = plt.subplot(413)
ay3.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
ay3.set_ylim([-3, 5])

ay4 = plt.subplot(414)
ay4.plot(residual, label='Residuals')
plt.legend(loc='best')
ay4.set_ylim([-3, 5])
plt.show()
plt.tight_layout

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(nh3_data['OUTPUT NH3-N'], model='additive',period=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

az1 = plt.subplot(411)
az1.plot(nh3_data['OUTPUT NH3-N'], label='Original')
plt.legend(loc='best')
az1.set_ylim([3, 14])

az1 = plt.subplot(412)
az1.plot(trend, label='Trend')
plt.legend(loc='best')
az1.set_ylim([3, 14])

az1 = plt.subplot(413)
az1.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
az1.set_ylim([-3, 5])

az1 = plt.subplot(414)
az1.plot(residual, label='Residuals')
plt.legend(loc='best')
az1.set_ylim([-3, 5])
plt.show()
plt.tight_layout