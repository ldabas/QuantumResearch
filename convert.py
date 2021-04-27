
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding
import numpy as np
import pandas as pd

data = pd.read_excel('./BOD Dataset.xlsx')
x = data[['NH3-N','BOD','TN','AT_Temp','MLSS']]
y = data['BOD_Y']

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(f=None):
    AmplitudeEmbedding(features=f, wires=range(2))
    return qml.expval(qml.PauliZ(0))

#a=x.iloc[0]
#print(a)
#C=([a[1],a[2],a[3],a[4]])
#print(C)

circuit(f=[1/2, 1/2, 1/2, 1/2])

