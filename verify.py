import pandas as pd
from pandas import read_csv
import numpy as np
from datetime import datetime
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

dfss_tn = dfss[['BOD', 'NH3-N', 'TN','PH']] #SSE dataset containing BOD, NH3, and TN values
dfat_tn=dfat[['MLSS','AT_Temp']]
dffe_tn = dffe[['TN']] #FE dataset containing NH3 values

dffe_tn.columns = ['OUTPUT TN']
tn_data = pd.concat([dfss_tn,dfat_tn,dffe_tn], axis=1)



def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=5).mean()
    rolstd = timeseries.rolling(window=5).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
test_stationarity(tn_data['OUTPUT TN'])

