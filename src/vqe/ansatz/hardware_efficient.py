"""Hardware-efficient (heuristic) ansatz."""

from qiskit.circuit.library import TwoLocal

def get_twolocal_ansatz(num_qubits, reps=1):
    """Build a TwoLocal ansatz: RY/RZ rotation layers with linear CZ entanglement.

    Args:
        num_qubits: Circuit width; must match the Hamiltonian.
        reps: Number of entangling layers.
    """
    return TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'linear', reps=reps)