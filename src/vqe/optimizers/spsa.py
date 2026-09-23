"""SPSA optimizer factory."""

from qiskit_algorithms.optimizers import SPSA

def get_spsa_optimizer(maxiter=200):
    """Return an SPSA optimizer.

    SPSA estimates the gradient from two energy evaluations per iteration,
    whatever the number of parameters, which suits noisy estimators. No seed
    is set here, so runs are not reproducible.
    """
    return SPSA(maxiter=maxiter)