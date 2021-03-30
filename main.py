import pennylane as qml
from pennylane import numpy as np
import sys

s3 = ("arn:aws:s3:::amazon-braket-72aea0b4ede6", "/")
remote_device = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=1)

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


def print_hi(name):
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


if __name__ == '__main__':
    # Load and process input
    angle_str = sys.stdin.read()
    angle = float(angle_str)

    ans = simple_circuits_20(angle)
    if isinstance(ans, np.tensor):
        ans = ans.item()

    if not isinstance(ans, float):
        raise TypeError("the simple_circuits_20 function needs to return a float")

    print(ans)

