import pennylane as qml
from pennylane import numpy as np
import sys
import boto3


#aws_account_id = boto3.client("sts").get_caller_identity()["ldabas-quantum"]
my_bucket = f"amazon-braket-72aea0b4ede6" # the name of the bucket#
my_prefix = "quantum-stuff/" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)
device_arn="arn:aws:braket:::device/qpu/ionq/ionQdevice"
wires = 1
remote_device = qml.device("braket.aws.qubit", device_arn = device_arn, s3_destination_folder=s3_folder, wires=wires)

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

