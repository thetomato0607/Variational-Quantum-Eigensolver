from qiskit.quantum_info import SparsePauliOp

def get_pauli_expectation(counts, pauli_op: SparsePauliOp):
    """
    Computes expectation value from counts for a specific Pauli string.
    Note: For VQE with StatevectorEstimator, this is often handled internally,
    but this function is useful for shot-based manual calculation.
    """
    # Placeholder for manual shot-based processing if needed in the future
    # Currently just returns 0.0 as we use Estimator primitives
    return 0.0

def filter_counts(counts, threshold=0):
    """Removes low-probability counts to clean up noise."""
    return {k: v for k, v in counts.items() if v > threshold}