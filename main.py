
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

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

dffe_tn = dffe[['TN']]
dffe_tn.columns = ['OUTPUT TN']
dffe_tnMA=dffe_tn.rolling(5, min_periods=1).mean()
dffe_tn.fillna(dffe_tnMA,inplace=True)

tn_data = pd.concat([dfss_fin,dfat_fin,dffe_tn], axis=1)
tn_data = tn_data.dropna()

features= tn_data[['BOD', 'NH3-N', 'TN','MLSS','PH','AT_Temp']]
labels= tn_data[['OUTPUT TN']]

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, y_train, x_test, y_test = train_test_split(features, labels, test_size = 0.20)


"""
aws_account_id = boto3.client("sts").get_caller_identity()["ldabas-quantum"]

my_bucket = f"amazon-braket-72aea0b4ede6" # the name of the bucket#
my_prefix = "quantum-stuff/" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)
device_arn="arn:aws:braket:::device/qpu/ionq/ionQdevice"

wires = 1
#remote_device = qml.device("braket.aws.qubit", device_arn = device_arn, s3_destination_folder=s3_folder, wires=wires)
remote_device = qml.device("default.qubit", wires = wires)
def simple_circuits_20(angle):

    @qml.qnode(remote_device)
    def my_first_fun(angle):
        qml.RX(angle, wires = 0)
        probs = qml.probs(0)
        return probs


    ans = my_first_fun(angle)
    prob = ans[0]
    prob = prob.item()



    return prob


if __name__ == '__main__':
    # Load and process input
    #angle_str = sys.stdin.read()
    angle = 35.5 #float(angle_str)

    ans = simple_circuits_20(angle)
    if isinstance(ans, np.tensor):
        ans = ans.item()

    if not isinstance(ans, float):
        raise TypeError("the simple_circuits_20 function needs to return a float")

    print(ans) """

