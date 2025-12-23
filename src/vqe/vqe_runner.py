"""
FINAL VQE LAB: FULL DISSOCIATION CURVE & ANSATZ COMPARISON
==========================================================
Fulfills Mindmap Layers 1, 4, and 7.
- Compares Physics-Inspired (UCCSD) vs Hardware-Efficient (TwoLocal)
- Scans bond length from 0.5A to 2.5A
- Uses Statevector for clean data (Layer 6.1)
"""
import numpy as np
import matplotlib.pyplot as plt

# Qiskit & Chemistry Imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

print("="*60)
print("VQE RESEARCH: ANSATZ COMPARISON & DISSOCIATION CURVE")
print("="*60)

# --- CONFIGURATION (Layer 1 & 7) ---
distances = [0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 2.5]  # The Geometry Scan
exact_energies = []
vqe_energies_twolocal = []
vqe_energies_uccsd = []

estimator = StatevectorEstimator()
optimizer = COBYLA(maxiter=100) # Layer 5.1

for d in distances:
    print(f"\n--- Bond Length: {d} Angstroms ---")
    
    # 1. SETUP SYSTEM (Layer 1)
    driver = PySCFDriver(atom=f"H 0.0 0.0 0.0; H 0.0 0.0 {d}", basis='sto-3g')
    problem = driver.run()
    H_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    nuc_rep = problem.nuclear_repulsion_energy
    
    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals
    mapper = JordanWignerMapper()

    # 2. EXACT SOLUTION (Layer 1.2 Reference)
    solver = NumPyMinimumEigensolver()
    result_exact = solver.compute_minimum_eigenvalue(H_op)
    E_exact = result_exact.eigenvalue.real + nuc_rep
    exact_energies.append(E_exact)
    print(f"  Exact Energy:    {E_exact:.5f} Ha")

    # 3. RUN VQE: TwoLocal (Layer 4.1 Hardware Efficient)
    ansatz_tl = TwoLocal(H_op.num_qubits, ['ry', 'rz'], 'cz', 'linear', reps=1)
    vqe_tl = VQE(estimator, ansatz_tl, optimizer)
    result_tl = vqe_tl.compute_minimum_eigenvalue(H_op)
    E_tl = result_tl.eigenvalue.real + nuc_rep
    vqe_energies_twolocal.append(E_tl)
    print(f"  TwoLocal Energy: {E_tl:.5f} Ha")

    # 4. RUN VQE: UCCSD (Layer 4.1 Chemistry Inspired)
    # Note: Requires Hartree-Fock Initialization
    init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    ansatz_uccsd = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=init_state)
    
    # We restrict iterations for speed, UCCSD is deeper!
    vqe_uccsd = VQE(estimator, ansatz_uccsd, optimizer) 
    result_uccsd = vqe_uccsd.compute_minimum_eigenvalue(H_op)
    E_uccsd = result_uccsd.eigenvalue.real + nuc_rep
    vqe_energies_uccsd.append(E_uccsd)
    print(f"  UCCSD Energy:    {E_uccsd:.5f} Ha")

# 5. VISUALIZATION (Layer 8 Results)
plt.figure(figsize=(10, 6))
plt.plot(distances, exact_energies, 'k--', label='Exact (FCI)')
plt.plot(distances, vqe_energies_uccsd, 'go-', label='UCCSD (Physics)')
plt.plot(distances, vqe_energies_twolocal, 'bo-', label='TwoLocal (Heuristic)')

plt.xlabel('Interatomic Distance (A)')
plt.ylabel('Energy (Hartree)')
plt.title('H2 Dissociation: Ansatz Comparison')
plt.legend()
plt.grid(True)
plt.savefig("final_dissociation_curve.png")
print("\n✓ Experiment Complete. Graph saved.")
plt.show()