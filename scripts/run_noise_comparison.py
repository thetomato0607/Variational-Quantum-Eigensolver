import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vqe.hamiltonians.h2 import get_h2_hamiltonian
from vqe.ansatz.hardware_efficient import get_twolocal_ansatz
from vqe.vqe_runner import VQERunner
from vqe.optimizers.scipy_opt import get_optimizer
from vqe.optimizers.spsa import get_spsa_optimizer
from vqe.backends.noisy import get_noisy_estimator
from qiskit_algorithms import NumPyMinimumEigensolver

print("🧪 Starting Noise Resilience Experiment...")
print("    Comparing COBYLA vs. SPSA on a noisy backend.")

# 1. Setup Physics (H2 at equilibrium)
distance = 0.735
print(f"\n--- System: H2 at {distance} Å ---")
qubit_op, problem = get_h2_hamiltonian(distance)
nuc_rep = problem.nuclear_repulsion_energy

# 2. Setup Noise Model (Depolarizing Noise)
noise_level = 0.02  # 2% error rate per gate (pretty high!)
print(f"--- Noise Level: {noise_level*100}% Depolarizing Error ---")
noisy_estimator = get_noisy_estimator(depolarizing_prob=noise_level)

# 3. Exact Solution (for reference)
solver = NumPyMinimumEigensolver()
result_exact = solver.compute_minimum_eigenvalue(qubit_op)
E_exact = result_exact.eigenvalue.real + nuc_rep
print(f"⚫ Exact Energy: {E_exact:.5f} Ha")

# 4. Run COBYLA (The "Standard" Optimizer)
print("\n🔵 Running COBYLA (Sensitive to noise)...")
# Decompose the ansatz into basic gates that Aer can handle
ansatz_base = get_twolocal_ansatz(qubit_op.num_qubits, reps=1)
ansatz = ansatz_base.decompose().decompose()  # Double decompose to get to basic gates
optimizer_cobyla = get_optimizer("COBYLA", maxiter=100)

runner_cobyla = VQERunner(qubit_op, ansatz, optimizer_cobyla, estimator=noisy_estimator)
result_cobyla = runner_cobyla.run()
E_cobyla = result_cobyla['optimal_value'] + nuc_rep
print(f"   COBYLA Final: {E_cobyla:.5f} Ha")

# 5. Run SPSA (The "Noise-Robust" Optimizer)
print("\n🟢 Running SPSA (Designed for noise)...")
# SPSA needs more iterations because it's stochastic, but each step is cheaper
optimizer_spsa = get_spsa_optimizer(maxiter=200) 

runner_spsa = VQERunner(qubit_op, ansatz, optimizer_spsa, estimator=noisy_estimator)
result_spsa = runner_spsa.run()
E_spsa = result_spsa['optimal_value'] + nuc_rep
print(f"   SPSA Final:   {E_spsa:.5f} Ha")

# 6. Plot Results
plt.figure(figsize=(10, 6))

# Plot history (Energies + Nuclear Repulsion to get Total Energy)
hist_cobyla = [e + nuc_rep for e in result_cobyla['history']]
hist_spsa = [e + nuc_rep for e in result_spsa['history']]

plt.plot(hist_cobyla, label='COBYLA (Noisy)', color='blue', alpha=0.6)
plt.plot(hist_spsa, label='SPSA (Noisy)', color='green', linewidth=2)
plt.axhline(E_exact, color='black', linestyle='--', label='Exact Energy')

plt.xlabel('Iterations')
plt.ylabel('Total Energy (Ha)')
plt.title(f'Optimizer Robustness: COBYLA vs SPSA (Noise={noise_level})')
plt.legend()
plt.grid(True, alpha=0.3)

save_path = 'results/h2/figures/noise_comparison.png'
plt.savefig(save_path)
print(f"\n✅ Graph saved to {save_path}")
print("   Observation: SPSA should look 'jumpy' but trend lower than COBYLA.")