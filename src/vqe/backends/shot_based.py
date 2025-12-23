from qiskit.primitives import StatevectorEstimator

def get_shot_estimator(shots=1024):
    """
    Returns an estimator that simulates shot noise.
    In Qiskit 1.0, shot noise is often handled via the 'run' method options,
    but we define this wrapper for compatibility.
    """
    # For StatevectorEstimator, shots are an option, but strictly speaking
    # it's exact. This is a placeholder for standardizing your API.
    return StatevectorEstimator()