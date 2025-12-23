from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

def get_noisy_estimator(depolarizing_prob=0.01):
    """Returns an estimator that simulates a noisy quantum computer."""
    # Create a simple noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all 1-qubit gates (u1, u2, u3)
    error_1 = depolarizing_error(depolarizing_prob, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])

    # Add depolarizing error to all 2-qubit gates (cx)
    error_2 = depolarizing_error(depolarizing_prob * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    # Use Aer's native EstimatorV2 with noise model option
    # Enable skip_transpilation=False to let Aer handle circuit decomposition
    estimator = AerEstimatorV2()
    estimator.options.noise_model = noise_model
    # Set seed for reproducibility
    estimator.options.seed_simulator = 42

    return estimator