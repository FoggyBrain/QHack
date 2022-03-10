#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.
    def rotY(theta):
        return [[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]]

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.

        for i in range(3):
            qml.Hadamard(wires = i)
            
        # qml.RY(thetas[0], wires = 3)
        for j in range(8):
            controls = np.zeros(3)
            i = 2
            val = j
            while val>0:
                controls[i]=val%2
                val = val // 2
                i -= 1
            ctrlval = ''.join([str(int(elem)) for elem in controls])
#             print(ctrlval)
            
            U = rotY(thetas[j])
#             print(U)
            qml.ControlledQubitUnitary(U, control_wires = range(3), wires = 3, control_values = ctrlval)



        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)
#     print(f"angles are: {thetas}")

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
