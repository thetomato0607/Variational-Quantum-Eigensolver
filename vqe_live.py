"""
VQE LIVE: REAL-TIME QUANTUM CHEMISTRY
=====================================
Uses PySCF to calculate H2 properties on the fly.
UPDATED FOR QISKIT 1.0+ / 2.0+
"""
import numpy as np
import matplotlib.pyplot as plt

# Core imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA

# FIXED: Use proper estimator based on Qiskit version
try:
    # Qiskit 1.0+ style
    from qiskit.primitives import StatevectorEstimator
    USE_NEW_API = True
except ImportError:
    # Fallback to older style
    from qiskit.primitives import Estimator
    USE_NEW_API = False

print("=" * 60)
print("VQE LIVE: CALCULATING H2 MOLECULE")
print("=" * 60)

# =============================================================================
# CONFIGURATION
# =============================================================================
BOND_DISTANCE = 0.735  # Angstroms
ANSATZ_DEPTH = 2
MAX_ITERATIONS = 100

print(f"\nConfiguration:")
print(f"  Bond distance:  {BOND_DISTANCE} Angstrom")
print(f"  Ansatz depth:   {ANSATZ_DEPTH}")
print(f"  Max iterations: {MAX_ITERATIONS}")

# =============================================================================
# STEP 1: QUANTUM CHEMISTRY CALCULATION (PYSCF)
# =============================================================================
print(f"\n1. Running PySCF for H2 at {BOND_DISTANCE} Angstroms...")

driver = PySCFDriver(
    atom=f"H 0.0 0.0 0.0; H 0.0 0.0 {BOND_DISTANCE}",
    charge=0,
    spin=0,
    basis='sto-3g'
)

# Run PySCF and get the problem
problem = driver.run()

# Extract Hamiltonian and map to qubits
hamiltonian_op = problem.hamiltonian.second_q_op()
mapper = JordanWignerMapper()
H_qubit = mapper.map(hamiltonian_op)

# Get nuclear repulsion
E_nuclear = problem.nuclear_repulsion_energy

print(f"   PySCF calculation complete")
print(f"   Number of qubits:     {H_qubit.num_qubits}")
print(f"   Pauli terms:          {len(H_qubit)}")
print(f"   Nuclear repulsion:    {E_nuclear:.8f} Ha")

# =============================================================================
# STEP 2: EXACT SOLUTION (CLASSICAL REFERENCE)
# =============================================================================
print("\n2. Calculating exact reference energy...")

exact_solver = NumPyMinimumEigensolver()
result_exact = exact_solver.compute_minimum_eigenvalue(H_qubit)
E_exact_electronic = result_exact.eigenvalue.real
E_exact_total = E_exact_electronic + E_nuclear

print(f"   Exact electronic:     {E_exact_electronic:.10f} Ha")
print(f"   Exact total:          {E_exact_total:.10f} Ha")

# =============================================================================
# STEP 3: VQE SETUP
# =============================================================================
print("\n3. Setting up VQE...")

# Build ansatz (parameterized quantum circuit)
ansatz = TwoLocal(
    num_qubits=H_qubit.num_qubits,
    rotation_blocks=['ry', 'rz'],
    entanglement_blocks='cx',
    entanglement='linear',
    reps=ANSATZ_DEPTH
)

print(f"   Ansatz parameters:    {ansatz.num_parameters}")
print(f"   Circuit depth:        {ansatz.decompose().depth()}")

# Set up optimizer
optimizer = COBYLA(maxiter=MAX_ITERATIONS)

# Set up estimator (handles Qiskit API differences)
if USE_NEW_API:
    estimator = StatevectorEstimator()
    print(f"   Using: StatevectorEstimator")
else:
    estimator = Estimator()
    print(f"   Using: Estimator")

# Tracking
convergence_history = []
iteration_count = [0]  # Use list to modify in callback

def progress_callback(eval_count, parameters, energy_mean, energy_std):
    """Track VQE progress at each iteration"""
    E_total = energy_mean + E_nuclear
    convergence_history.append(E_total)
    iteration_count[0] = eval_count
    
    if eval_count % 10 == 0 or eval_count == 1:
        error = E_total - E_exact_total
        print(f"   Iteration {eval_count:3d}: E = {E_total:.8f} Ha, "
              f"Error = {error:+.6f} Ha")

# =============================================================================
# STEP 4: RUN VQE OPTIMIZATION
# =============================================================================
print("\n4. Running VQE optimization...")
print("   (This may take 1-3 minutes)\n")

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=progress_callback
)

result_vqe = vqe.compute_minimum_eigenvalue(H_qubit)

# Extract results
E_vqe_electronic = result_vqe.eigenvalue.real
E_vqe_total = E_vqe_electronic + E_nuclear
error_total = E_vqe_total - E_exact_total
error_kcal = error_total * 627.5

# =============================================================================
# STEP 5: RESULTS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\nEnergies (Hartree):")
print(f"  VQE total energy:     {E_vqe_total:.10f} Ha")
print(f"  Exact total energy:   {E_exact_total:.10f} Ha")
print(f"  Absolute error:       {abs(error_total):.10f} Ha")
print(f"  Error (kcal/mol):     {abs(error_kcal):.4f} kcal/mol")

print(f"\nPerformance:")
print(f"  Total iterations:     {iteration_count[0]}")
print(f"  Optimal parameters:   {len(result_vqe.optimal_point)}")

# Chemical accuracy check
CHEM_ACCURACY = 0.0016  # 1 kcal/mol
if abs(error_total) < CHEM_ACCURACY:
    print(f"\nChemical accuracy:    Achieved")
else:
    print(f"\nChemical accuracy:    Not achieved")
    print(f"  Target:               < {CHEM_ACCURACY:.6f} Ha")
    print(f"  Current:              {abs(error_total):.6f} Ha")

# =============================================================================
# STEP 6: VISUALIZATION
# =============================================================================
print("\n5. Generating convergence plot...")

plt.figure(figsize=(10, 6))

# Plot VQE convergence
iterations = list(range(1, len(convergence_history) + 1))
plt.plot(iterations, convergence_history, 'b-', linewidth=2, label='VQE Energy')

# Plot exact reference
plt.axhline(y=E_exact_total, color='r', linestyle='--', linewidth=2, 
            label='Exact Energy')

# Chemical accuracy band
plt.fill_between(iterations, 
                 E_exact_total - CHEM_ACCURACY, 
                 E_exact_total + CHEM_ACCURACY,
                 color='green', alpha=0.2, 
                 label='Chemical Accuracy (±1 kcal/mol)')

# Labels and styling
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Total Energy (Hartree)', fontsize=12)
plt.title(f'VQE Convergence: H₂ at {BOND_DISTANCE} Å (Depth={ANSATZ_DEPTH})', 
          fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Save figure
filename = f"vqe_live_{BOND_DISTANCE}A.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"   Plot saved: {filename}")

plt.show()

# =============================================================================
# STEP 7: DATA EXPORT
# =============================================================================
print("\n6. Exporting data...")

# Save convergence data
import csv

csv_filename = f"vqe_convergence_{BOND_DISTANCE}A.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Energy_Ha', 'Error_Ha', 'Error_kcal_mol'])
    for i, energy in enumerate(convergence_history, 1):
        err = energy - E_exact_total
        writer.writerow([i, energy, err, err * 627.5])

print(f"   Data saved: {csv_filename}")

# Save summary
summary_filename = f"vqe_summary_{BOND_DISTANCE}A.txt"
with open(summary_filename, 'w') as f:
    f.write("VQE CALCULATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"System: H2 molecule\n")
    f.write(f"Bond distance: {BOND_DISTANCE} Angstrom\n")
    f.write(f"Basis set: STO-3G\n\n")
    f.write(f"VQE Configuration:\n")
    f.write(f"  Ansatz depth:     {ANSATZ_DEPTH}\n")
    f.write(f"  Parameters:       {ansatz.num_parameters}\n")
    f.write(f"  Optimizer:        COBYLA\n")
    f.write(f"  Max iterations:   {MAX_ITERATIONS}\n\n")
    f.write(f"Results:\n")
    f.write(f"  VQE energy:       {E_vqe_total:.10f} Ha\n")
    f.write(f"  Exact energy:     {E_exact_total:.10f} Ha\n")
    f.write(f"  Error:            {error_total:+.10f} Ha\n")
    f.write(f"  Error (kcal/mol): {error_kcal:+.4f} kcal/mol\n")
    f.write(f"  Iterations:       {iteration_count[0]}\n")

print(f"   Summary saved: {summary_filename}")

# =============================================================================
# COMPLETION
# =============================================================================
print("\n" + "=" * 60)
print("VQE CALCULATION COMPLETE")
print("=" * 60)

print(f"\nFiles generated:")
print(f"  1. {filename}")
print(f"  2. {csv_filename}")
print(f"  3. {summary_filename}")

print(f"\nFinal result:")
print(f"  VQE:   {E_vqe_total:.8f} Ha")
print(f"  Exact: {E_exact_total:.8f} Ha")
print(f"  Error: {abs(error_total):.8f} Ha ({abs(error_kcal):.3f} kcal/mol)")

print("\n" + "=" * 60)