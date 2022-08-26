
#### NOTE: This file has been copied and renamed.
#### Copied from https://github.com/lanl-ansi/max-cut/tree/456d8c23251bacc8d6b132e918fa4543e5de42f8
#### Original Name of file: solve-qaoa-long.py

#!/usr/bin/env python3

import os, statistics, math, random

from collections import namedtuple

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit import Aer

import common


# based on examples from https://qiskit.org/textbook/ch-applications/qaoa.html

QAOA_Parameter  = namedtuple('QAOA_Parameter', ['beta', 'gamma'])
BACKEND = Aer.get_backend('qasm_simulator')


def _create_qaoa_circ(G, parameters):
    nqubits, edges = G

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for par in parameters:
        # problem unitary
        for i,j in edges:
            qc.rzz(2 * par.gamma, i, j)

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    qc.measure_all()

    return qc

def _compute_statistics(counts, G):
    nodes, edges = G

    solutions = []
    for solution, count in counts.items():
        #print(count)
        sol_eval = -1*common.eval_cut(nodes, edges, solution)
        for i in range(count):
            solutions.append(sol_eval)

    # for i,sol in enumerate(solutions):
    #     print(i, " ", sol)

    eval_max = min(solutions)
    eval_mean = statistics.mean(solutions)
    eval_sd = statistics.stdev(solutions)
    eval_quant = statistics.quantiles(solutions, n=4)

    #print()
    #print("  maximum cut.: {:.2f}".format(eval_max))
    #print("  expected cut: {:.2f}".format(eval_mean))
    #print("  sd cut......: {:.2f}".format(eval_sd))

    return (eval_max, eval_mean, eval_sd, eval_quant)

def execute_circ(G, theta, shots):
    p = len(theta)//2  # number of qaoa rounds
    beta = theta[:p]
    gamma = theta[p:]
    parameters = [QAOA_Parameter(*t) for t in zip(beta,gamma)]

    qc = _create_qaoa_circ(G, parameters)

    result = BACKEND.run(qc, seed_simulator=10, shots=shots).result()
    counts = result.get_counts()

    return _compute_statistics(counts, G)


def compute_expectation(G, shots):

    def execute_circ_closure(theta):
        return execute_circ(G, theta, shots)[1]

    return execute_circ_closure


def maxcut_qaoa(nodes, edges, shots, rounds=1):
    expectation_func = compute_expectation((nodes,edges), shots)

    res_best = None
    for i in range(100):
        print("iteration: {}".format(i))

        # starting point higligts instability of the optimization method
        theta = [random.uniform(-math.pi, math.pi) for i in range(2*rounds)]
        bounds = [(-math.pi, math.pi) for i in range(2*rounds)]
        #theta = 2*rounds*[0.1]
        #theta = 2*rounds*[0.0]

        res = minimize(expectation_func, theta, bounds=bounds, method='COBYLA')
        #res = minimize(expectation_func, theta, bounds=bounds, method='BFGS')
        #res = minimize(expectation_func, theta, bounds=bounds, method='CG')
        #print(res)

        if res_best == None or res.fun < res_best.fun:
            print("update res best: {}".format(res.fun))
            print(res)
            
            res_best = res

    theta_best = res_best.x
    results = execute_circ((nodes, edges), theta_best, shots)

    results[3].reverse()
    return (-results[0], -results[1], results[2], *[-v for v in results[3]])
