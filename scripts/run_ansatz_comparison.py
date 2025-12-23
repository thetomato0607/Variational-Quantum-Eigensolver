import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vqe.hamiltonians.h2 import get_h2_hamiltonian
from vqe.ansatz.hardware_efficient import get_twolocal_ansatz
from vqe.ansatz.ucc_like import get_uccsd_ansatz
from vqe.vqe_runner import VQERunner
from vqe.optimizers.scipy_opt import get_optimizer

print("🧪 Starting Ansatz Comparison Experiment...")

# We will look at just one difficult point (stretched bond)
# At equilibrium (0.735), both are easy. At 1.5 or 2.0, it gets harder!
distance = 1.5
print(f"--- Analyzing H2 at Bond Length: {distance} Å ---")

# 1. Get Physics
qubit_op, problem = get_h2_hamiltonian(distance)
nuc_rep = problem.nuclear_repulsion_energy

# 2. Run TwoLocal (Heuristic)
print("\n🔵 Running TwoLocal (Hardware Efficient)...")
ansatz_tl = get_twolocal_ansatz(qubit_op.num_qubits, reps=1)
optimizer = get_optimizer("COBYLA", maxiter=200)

runner_tl = VQERunner(qubit_op, ansatz_tl, optimizer)
result_tl = runner_tl.run()
E_tl = result_tl['optimal_value'] + nuc_rep
print(f"   TwoLocal Energy: {E_tl:.5f} Ha")

# 3. Run UCCSD (Chemistry Inspired)
print("\n🟢 Running UCCSD (Physics Inspired)...")
ansatz_ucc = get_uccsd_ansatz(problem)
# UCCSD is harder to optimize, so we give it SPSA or just more COBYLA steps
optimizer_ucc = get_optimizer("COBYLA", maxiter=200)

runner_ucc = VQERunner(qubit_op, ansatz_ucc, optimizer_ucc)
result_ucc = runner_ucc.run()
E_ucc = result_ucc['optimal_value'] + nuc_rep
print(f"   UCCSD Energy:    {E_ucc:.5f} Ha")

# 4. Compare with Exact
from qiskit_algorithms import NumPyMinimumEigensolver
solver = NumPyMinimumEigensolver()
result_exact = solver.compute_minimum_eigenvalue(qubit_op)
E_exact = result_exact.eigenvalue.real + nuc_rep
print(f"\n⚫ Exact Energy:    {E_exact:.5f} Ha")

# 5. Plot Convergence Comparison
plt.figure(figsize=(10, 6))
plt.plot(result_tl['history'], label='TwoLocal', color='blue', alpha=0.7)
plt.plot(result_ucc['history'], label='UCCSD', color='green', alpha=0.7)
plt.axhline(result_exact.eigenvalue.real, color='black', linestyle='--', label='Exact Electronic Energy')

plt.xlabel('Iterations')
plt.ylabel('Electronic Energy (Ha)')
plt.title(f'Convergence Comparison (H2 at {distance}Å)')
plt.legend()
plt.grid(True, alpha=0.3)
save_path = 'results/h2/figures/ansatz_comparison.png'
plt.savefig(save_path)
print(f"Comparison graph saved to {save_path}")