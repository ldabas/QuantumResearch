import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from IPython.display import display

# Works if Dataset.xlsx is uploaded directly on filesys on left
with pd.ExcelFile('./Dataset.xlsx') as reader:
    # Train from data
    sheet1 = pd.read_excel(reader, sheet_name='SS(Ave)')[['Date', 'BOD', 'NH3-N', 'TN', 'PH']]
    sheet3 = pd.read_excel(reader, sheet_name='AT(Ave)')[['Date', 'MLSS', 'AT_Temp']]
    # test data
    sheet2 = pd.read_excel(reader, sheet_name='FE')[['Date', 'BOD', 'NH3-N', 'TN']]

# Make Date the index
sheet1.set_index('Date', inplace=True)
sheet2.set_index('Date', inplace=True)
sheet3.set_index('Date', inplace=True)

df_inputs = pd.merge(sheet1, sheet3, on='Date', how='outer')
df_outputs = sheet2

df_inputs

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler, IterativeImputer

df_inputs = pd.DataFrame(data=IterativeImputer().fit_transform(df_inputs), columns=df_inputs.columns, index=df_inputs.index)
df_outputs = pd.DataFrame(data=IterativeImputer().fit_transform(df_outputs), columns=df_outputs.columns, index=df_outputs.index)
df_inputs.describe()

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for Ammonia')
    plt.show()

#test_stationarity(df_outputs['TN'])
#test_stationarity(df_outputs['BOD'])
test_stationarity(df_outputs['NH3-N'])

def difference(dataset, interval=1):
    index = list(dataset.index)
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset["TN"][i] - dataset["TN"][i - interval]
        diff.append(value)
    return (diff)

diff = difference(df_outputs)
plt.plot(diff)
plt.show()

tn_log = np.log(df_outputs['TN'])
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

fig = plt.figure()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df_outputs['TN'], model='additive',period=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

ax1 = plt.subplot(411)
ax1.plot(df_outputs['TN'], label='Original')
plt.legend(loc='best')
plt.title('Time Series Forecasting: Total Nitrogen')
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
display(fig)


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df_outputs['BOD'], model='additive',period=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

ay1 = plt.subplot(411)
ay1.plot(df_outputs['BOD'], label='Original')
plt.legend(loc='best')
plt.title('Time Series Forecasting: Biological Oxygen Demand')
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


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df_outputs['NH3-N'], model='additive',period=30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

az1 = plt.subplot(411)
az1.plot(df_outputs['NH3-N'], label='Original')
plt.title('Time Series Forecasting: Ammonia')
plt.legend(loc='best')
az1.set_ylim([0, 2])

az1 = plt.subplot(412)
az1.plot(trend, label='Trend')
plt.legend(loc='best')
az1.set_ylim([0, 3])

az1 = plt.subplot(413)
az1.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
az1.set_ylim([-3, 5])

az1 = plt.subplot(414)
az1.plot(residual, label='Residuals')
plt.legend(loc='best')
az1.set_ylim([-3, 5])
plt.show()