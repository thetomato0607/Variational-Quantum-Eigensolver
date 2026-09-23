"""Transverse-field Ising model (TFIM) Hamiltonian."""

from qiskit.quantum_info import SparsePauliOp

def get_tfim_hamiltonian(num_qubits: int, j_coupling: float = 1.0, g_field: float = 1.0):
    """Build the TFIM chain Hamiltonian H = -J * sum(Z_i Z_i+1) - g * sum(X_i).

    Uses open boundary conditions: there is no coupling between the last and
    first qubit.
    """
    pauli_list = []
    
    # Interaction term (ZZ)
    for i in range(num_qubits - 1):
        pauli_str = ["I"] * num_qubits
        pauli_str[i] = "Z"
        pauli_str[i+1] = "Z"
        # Qiskit orders Pauli labels with qubit 0 rightmost, hence the reversal.
        pauli_list.append(("".join(pauli_str)[::-1], -j_coupling))
        
    # Transverse field term (X)
    for i in range(num_qubits):
        pauli_str = ["I"] * num_qubits
        pauli_str[i] = "X"
        pauli_list.append(("".join(pauli_str)[::-1], -g_field))
        
    return SparsePauliOp.from_list(pauli_list)