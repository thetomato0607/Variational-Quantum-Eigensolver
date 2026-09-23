"""Shot-based estimator backend (placeholder)."""

from qiskit.primitives import StatevectorEstimator

def get_shot_estimator(shots=1024):
    """Return a ``StatevectorEstimator``; ``shots`` is currently ignored.

    Despite the name, the result is exact: no shot noise is simulated. It is
    also a V2 estimator, which qiskit-algorithms 0.3.1's ``VQE`` does not accept.
    """
    return StatevectorEstimator()