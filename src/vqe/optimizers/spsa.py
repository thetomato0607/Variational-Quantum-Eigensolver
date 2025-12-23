from qiskit_algorithms.optimizers import SPSA

def get_spsa_optimizer(maxiter=200):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA).
    Ideal for noisy VQE because it approximates the gradient with only 2 measurements.
    """
    return SPSA(maxiter=maxiter)