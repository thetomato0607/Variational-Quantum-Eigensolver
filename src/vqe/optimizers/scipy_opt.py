from qiskit_algorithms.optimizers import COBYLA

def get_optimizer(name="COBYLA", maxiter=200):
    return COBYLA(maxiter=maxiter)