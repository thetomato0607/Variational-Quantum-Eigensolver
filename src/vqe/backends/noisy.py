"""Depolarizing-noise estimator backend (local Aer simulation, not hardware)."""

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

def get_noisy_estimator(depolarizing_prob=0.01):
    """Return an Aer ``EstimatorV2`` with a uniform depolarizing noise model.

    Args:
        depolarizing_prob: Single-qubit error probability; two-qubit (CX) gates
            use 10x this value.

    Note:
        This is a V2 estimator. qiskit-algorithms 0.3.1's ``VQE`` (used by
        ``VQERunner``) expects the V1 interface.
    """
    noise_model = NoiseModel()

    # Errors attach by gate name, so gates missing from these lists (e.g. r, h,
    # ry, cz) run noise-free unless the circuit is first transpiled to them.
    error_1 = depolarizing_error(depolarizing_prob, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])

    # Assumption: two-qubit gates are 10x noisier than single-qubit gates.
    error_2 = depolarizing_error(depolarizing_prob * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    estimator = AerEstimatorV2()
    estimator.options.noise_model = noise_model
    # Fixed seed: the only seeded source of randomness in the pipeline.
    estimator.options.seed_simulator = 42

    return estimator