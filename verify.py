import pandas as pd

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