import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

qubit = cirq.GridQubit(0, 0)

# Define some circuits.
circuit1 = cirq.Circuit(cirq.X(qubit))
circuit2 = cirq.Circuit(cirq.H(qubit))

# Convert to a tensor.
input_circuit_tensor = tfq.convert_to_tensor([circuit1, circuit2])

# Define a circuit that we want to append
y_circuit = cirq.Circuit(cirq.Y(qubit))

# Instantiate our layer
y_appender = tfq.layers.AddCircuit()

# Run our circuit tensor through the layer and save the output.
output_circuit_tensor = y_appender(input_circuit_tensor, append=y_circuit)

#print(tfq.from_tensor(input_circuit_tensor))

#print(tfq.from_tensor(output_circuit_tensor))

#Get Data

data = pd.read_excel('Dataset.xlsx', sheet_name=['SS(Ave)', 'FE','AT(Ave)'])
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


dfss_tnMA = dfss_tn.rolling(5, min_periods=1).mean()
dfat_tnMA=dfat_tn.rolling(5, min_periods=1).mean()
dffe_tnMA=dffe_tn.rolling(5, min_periods=1).mean()

dfss_tn.fillna(dfss_tnMA,inplace=True)
dfat_tn.fillna(dfat_tnMA,inplace=True)
dffe_tn.fillna(dffe_tnMA,inplace=True)

dffe_tn.columns = ['OUTPUT TN']
tn_data = pd.concat([dfss_tn,dfat_tn,dffe_tn], axis=1)

tn_data = tn_data.dropna()

from sklearn.model_selection import train_test_split

def generate_data(data):
    features = data[['BOD', 'NH3-N', 'TN', 'MLSS', 'PH', 'AT_Temp']]
    labels = data[['OUTPUT TN']]
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
    return train_features, test_features, train_labels, test_labels


train_features, test_features, train_labels, test_labels = generate_data(tn_data)#FEATURES->X AND LABELS ->Y


def convert_to_circuit(row):
    qubits = cirq.GridQubit.rect(4,4)
    circuit = cirq.Circuit()
    for i,x in enumerate(row):
        if x:
            circuit.append(cirq.X(qubits[i]))
    return circuit


q_train_features = [convert_to_circuit(row) for row in train_features]
q_test_features = [convert_to_circuit(row) for row in test_features]

features_train_tf = tfq.convert_to_tensor(q_train_features)
features_test_tf = tfq.convert_to_tensor(q_test_features)


class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout) ** symbol)

demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(4,1),
                                   readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate = cirq.XX, prefix='xx')


def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)  # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

model_circuit, model_readout = create_quantum_model()


# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),
])

#y_train_hinge = 2.0*train_labels-1.0
#y_test_hinge = 2.0*test_labels-1.0