#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    wires = cnot.wires
    start = wires[0]
    end = wires[1]
    
    dist = np.full(9,np.inf)
    dist[start]=0
    unvisited = set([0,1,2,3,4,5,6,7,8])
    current = start
    
    while True:
        for neighbor in graph[current]:
            if neighbor in unvisited:
                dist[neighbor]=min(dist[neighbor],dist[current]+1)
        unvisited.remove(current)
        # print(unvisited)
        mindis_unvis = np.inf
        for i in unvisited:
            if dist[i]<mindis_unvis:
                current = i
                mindis_unvis=dist[i]
        if current == end:
            break
    
    return 2*(int(dist[current])-1)
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
