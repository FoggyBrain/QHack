#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    theta = np.arctan(beta/alpha)
    qml.RY(theta*2, wires = 0)
    qml.CNOT(wires = [0,1])
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    if x == 0:
        thetaA = theta_A0
    else:
        thetaA = theta_A1       
    if y == 0:
        thetaB = theta_B0
    else:
        thetaB = theta_B1
        
    qml.RY(thetaA*2, wires = 0)
    qml.RY(thetaB*2, wires = 1)

    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    
    probs = []
    for x in range(2):
        for y in range(2):
            answers = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)
            if x*y==0:
                probs.append(answers[0]+answers[3])
            else:
                probs.append(answers[1]+answers[2])
#     print('computed winning probs')            
    return sum(probs)/len(probs)
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
#         print('computing cost')
        return (1-winning_prob(params, alpha, beta)**2)

    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    np.random.seed(0)
    init_params = np.pi*np.random.randn(4, requires_grad = True)
    opt = qml.optimize.AdamOptimizer(0.5)
    steps =300

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #

        params, cost_f = opt.step_and_cost(cost, params)
        
        if np.abs(cost(params)-cost_f)<1e-6:
            break
    
#         if i%10 == 0:
#             print(f'step {i}: winning_probs = {winning_prob(params, alpha, beta)}|cost_fn = {cost_f}')
        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")