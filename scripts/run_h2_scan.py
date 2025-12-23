import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vqe.hamiltonians.h2 import get_h2_hamiltonian
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

print("Starting H2 Dissociation Scan...")

distances = [0.5, 0.735, 1.0, 1.5, 2.0, 2.5]
vqe_energies = []
exact_energies = []

estimator = StatevectorEstimator()
optimizer = COBYLA(maxiter=100)

for d in distances:
    print(f"\n--- Distance: {d} Å ---")
    
    # 1. Get Physics (Using your new module!)
    H_op, nuc_rep, _, _ = get_h2_hamiltonian(d)
    
    # 2. Exact Solution
    solver = NumPyMinimumEigensolver()
    result_exact = solver.compute_minimum_eigenvalue(H_op)
    E_exact = result_exact.eigenvalue.real + nuc_rep
    exact_energies.append(E_exact)
    
    # 3. VQE
    ansatz = TwoLocal(H_op.num_qubits, ['ry', 'rz'], 'cz', 'linear', reps=1)
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(H_op)
    E_vqe = result.eigenvalue.real + nuc_rep
    vqe_energies.append(E_vqe)
    
    print(f"   VQE: {E_vqe:.5f} Ha")

# Plotting
plt.plot(distances, exact_energies, 'k--', label='Exact')
plt.plot(distances, vqe_energies, 'bo-', label='VQE')
plt.xlabel('Distance (Å)')
plt.ylabel('Energy (Ha)')
plt.legend()
plt.savefig('results/h2/figures/dissociation_curve.png')
print("Graph saved to results/h2/figures/dissociation_curve.png")