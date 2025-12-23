from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator

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
    
    # Create the simulator
    backend = AerSimulator(noise_model=noise_model)
    
    # Return a primitive that uses this noisy backend
    return BackendEstimator(backend=backend)