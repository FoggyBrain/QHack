#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    ftype = np.zeros(4)
    dev = qml.device('default.qubit', wires = 3, shots = 1)
    
    @qml.qnode(dev)
    def circuit(i, oracle, wires):
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #

        # Insert any pre-oracle processing here
        for j in range(i%2):
            qml.PauliX(wires = 1)
        for j in range(i//2):
            qml.PauliX(wires = 0)
        
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.PauliX(wires=2)
        qml.Hadamard(wires=2)

        oracle(wires)  # DO NOT MODIFY this line

        # Insert any post-oracle processing here
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        # QHACK #

        return qml.sample(wires=range(2))
    
    for i in range(4):       
        output = circuit(i, fs[i], [0,1,2])
#         print(output)
        if output[0]*2+output[1]==i:
            ftype[i]=1
        dev.reset()
#         print(ftype[i])
    
    if sum(ftype)%4==0:
        return "4 same"
    
    return "2 and 2"
    
    # return 'end'
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
