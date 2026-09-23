"""Gradient-free optimizer factory."""

from qiskit_algorithms.optimizers import COBYLA

def get_optimizer(name="COBYLA", maxiter=200):
    """Return a COBYLA optimizer.

    ``name`` is ignored: COBYLA is always returned, whatever is passed.
    """
    return COBYLA(maxiter=maxiter)