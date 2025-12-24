import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vqe.hamiltonians.tfim import get_tfim_hamiltonian
from vqe.ansatz.hardware_efficient import get_twolocal_ansatz
from vqe.vqe_runner import VQERunner
from vqe.optimizers.scipy_opt import get_optimizer
from qiskit_algorithms import NumPyMinimumEigensolver

print("🧲 Starting TFIM Field Scan...")

# 1. Experiment Parameters
num_qubits = 4
g_values = np.linspace(0.1, 2.0, 10)  # Scan field g from 0.1 to 2.0
vqe_energies = []
exact_energies = []

print(f"--- System: {num_qubits}-Qubit Chain ---")
print(f"--- Scanning g-field: {g_values[0]} -> {g_values[-1]} ---")

# 2. The Scan Loop
for g in g_values:
    # A. Get Physics (J=1.0 is standard ferromagnetic coupling)
    qubit_op = get_tfim_hamiltonian(num_qubits, j_coupling=1.0, g_field=g)
    
    # B. Exact Solution
    solver = NumPyMinimumEigensolver()
    result_exact = solver.compute_minimum_eigenvalue(qubit_op)
    E_exact = result_exact.eigenvalue.real
    exact_energies.append(E_exact)
    
    # C. VQE Run
    # We use a shallow ansatz because TFIM is relatively simple
    ansatz = get_twolocal_ansatz(num_qubits, reps=1)
    optimizer = get_optimizer("COBYLA", maxiter=100)
    
    runner = VQERunner(qubit_op, ansatz, optimizer, verbose=False)
    result_vqe = runner.run()
    E_vqe = result_vqe['optimal_value']
    vqe_energies.append(E_vqe)
    
    print(f"   g={g:.2f}: Exact={E_exact:.4f}, VQE={E_vqe:.4f}")

# 3. Plotting
plt.figure(figsize=(10, 6))
plt.plot(g_values, exact_energies, 'k--', label='Exact (FCI)')
plt.plot(g_values, vqe_energies, 'ro-', label='VQE (TwoLocal)', alpha=0.8)

plt.xlabel('Transverse Field Strength (g)')
plt.ylabel('Ground State Energy (Ha)')
plt.title(f'TFIM Phase Diagram ({num_qubits} Qubits)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save figure
output_path = 'results/tfim/figures/tfim_scan.png'
plt.savefig(output_path)
print(f"\n✅ Graph saved to {output_path}")