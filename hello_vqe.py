from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
import numpy as np

import warnings
warnings.filterwarnings('ignore')

print("--- Starting VQE Simulation for H2 Molecule ---")

# 1. DEFINE THE MOLECULE
driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis='sto-3g',
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM
)
problem = driver.run()
print("1. Molecule defined. Orbitals calculated.")

# 2. MAP TO QUBITS
hamiltonian = problem.hamiltonian.second_q_op()
mapper = JordanWignerMapper()
qubit_op = mapper.map(hamiltonian)
print("2. Math converted to Qubit Operators.")

# 3. SET UP THE GUESSER (ANSATZ)
ansatz = TwoLocal(num_qubits=qubit_op.num_qubits, rotation_blocks='ry', entanglement_blocks='cz')

# 4. SET UP THE OPTIMIZER
optimizer = COBYLA(maxiter=100)

# 5. RUN VQE
print("3. Running VQE loop... (this might take a few seconds)")

# --- FIX IS HERE: Use StatevectorEstimator (V2) ---
# This is an exact simulator, so it's very fast and accurate.
estimator = StatevectorEstimator()

vqe = VQE(estimator, ansatz, optimizer)
result = vqe.compute_minimum_eigenvalue(qubit_op)

print("\n--- RESULTS ---")
print(f"VQE Computed Energy: {result.eigenvalue.real:.5f} Hartrees")
print(f"Reference Energy:    {problem.reference_energy:.5f} Hartrees")
print("Success! You just simulated a molecule.")