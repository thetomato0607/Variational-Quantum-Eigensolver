from qiskit.circuit.library import TwoLocal

def get_twolocal_ansatz(num_qubits, reps=1):
    return TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'linear', reps=reps)