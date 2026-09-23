"""Counts-based measurement helpers (not used by the current pipeline)."""

from qiskit.quantum_info import SparsePauliOp

def get_pauli_expectation(counts, pauli_op: SparsePauliOp):
    """Placeholder for a counts-based expectation value; always returns 0.0.

    The pipeline uses Estimator primitives, so nothing calls this yet.
    """
    return 0.0

def filter_counts(counts, threshold=0):
    """Drop bitstrings with ``threshold`` or fewer counts."""
    return {k: v for k, v in counts.items() if v > threshold}