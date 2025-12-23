"""
VQE Implementation: H2 Molecular Ground State Calculation
=========================================================

Implementation of the Variational Quantum Eigensolver algorithm for
determining the ground state electronic energy of the hydrogen molecule.

Features:
- Exact statevector simulation with optional shot noise
- Variance-based noise modeling for NISQ device simulation
- Multi-trial statistical analysis
- Complete parameter trajectory tracking
- Comprehensive data export for post-processing

Author: Research Implementation
Date: 2024
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from datetime import datetime
from pathlib import Path

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP

# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

# Circuit Architecture
ANSATZ_DEPTH = 1           # Number of ansatz repetitions
ENTANGLEMENT = 'linear'    # Qubit connectivity topology
USE_HF_INIT = True         # Initialize from Hartree-Fock reference state

# Optimization Configuration  
OPTIMIZER = 'COBYLA'       # Classical optimization algorithm
MAX_ITERATIONS = 200       # Maximum optimizer function evaluations

# Measurement Simulation
SHOTS = 4096            # Measurement shots (None = exact expectation values)

# Statistical Analysis
RUN_MULTIPLE_TRIALS = False  # Execute multiple independent optimization runs
NUM_TRIALS = 5              # Number of trials for statistical analysis

# Output Configuration
SAVE_PLOTS = True          # Automatically save generated figures
VERBOSE = True             # Print iteration-level progress

print("=" * 80)
print("VQE IMPLEMENTATION: H2 MOLECULAR SYSTEM")
print("=" * 80)
print("\nExperimental Configuration:")
print(f"  Ansatz depth:        {ANSATZ_DEPTH}")
print(f"  Entanglement:        {ENTANGLEMENT}")
print(f"  Initial state:       {'Hartree-Fock' if USE_HF_INIT else 'Vacuum'}")
print(f"  Optimizer:           {OPTIMIZER}")
print(f"  Max iterations:      {MAX_ITERATIONS}")
print(f"  Shot noise:          {f'{SHOTS} shots' if SHOTS else 'Disabled'}")
print(f"  Statistical trials:  {NUM_TRIALS if RUN_MULTIPLE_TRIALS else 1}")

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================
print("\n" + "-" * 80)
print("Loading Molecular Hamiltonian")
print("-" * 80)

try:
    with open("flight_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Reconstruct qubit Hamiltonian from Pauli decomposition
    H_operator = SparsePauliOp.from_list(data["pauli_list"])
    
    # Extract reference energies - handle both possible key names
    E_nuclear = data.get("nuclear_repulsion") or data.get("nuclear_repulsion_energy")
    E_reference_total = data.get("exact_total_energy") or data.get("total_energy")
    E_reference_electronic = data.get("exact_electronic_energy") or data.get("electronic_energy")
    
    # Validate critical parameters
    if E_nuclear is None:
        raise KeyError("Nuclear repulsion energy not found in data file")
    if E_reference_total is None:
        raise KeyError("Total reference energy not found in data file")
    if E_reference_electronic is None:
        # Calculate from total if not provided
        E_reference_electronic = E_reference_total - E_nuclear
    
    print(f"Molecular system:      {data.get('molecule', 'H2')}")
    print(f"Bond length:           {data.get('bond_length', data.get('bond_length_angstrom', 'N/A'))} Angstrom")
    print(f"Qubit count:           {H_operator.num_qubits}")
    print(f"Hamiltonian terms:     {len(data['pauli_list'])}")
    print(f"Nuclear repulsion:     {E_nuclear:.10f} Ha")
    print(f"Reference total:       {E_reference_total:.10f} Ha")
    print(f"Reference electronic:  {E_reference_electronic:.10f} Ha")
    
except FileNotFoundError:
    print("ERROR: Required data file 'flight_data.pkl' not found.")
    print("Execute data generation script before running VQE.")
    exit(1)
except KeyError as e:
    print(f"ERROR: Missing required data field: {e}")
    print("\nAvailable keys in data file:")
    try:
        with open("flight_data.pkl", "rb") as f:
            data_check = pickle.load(f)
            for key in data_check.keys():
                print(f"  - {key}")
    except:
        pass
    exit(1)

# =============================================================================
# CLASSICAL VERIFICATION
# =============================================================================
print("\n" + "-" * 80)
print("Classical Verification via Exact Diagonalization")
print("-" * 80)

H_matrix = H_operator.to_matrix()
eigenvalues = np.linalg.eigvalsh(H_matrix)
E_classical_ground = eigenvalues[0]
E_classical_total = E_classical_ground + E_nuclear

print(f"Classical ground state (electronic): {E_classical_ground:.10f} Ha")
print(f"Classical ground state (total):      {E_classical_total:.10f} Ha")

# Verification check
energy_match = abs(E_classical_total - E_reference_total) < 1e-8
print(f"Reference verification:              {energy_match}")

if not energy_match:
    print("WARNING: Classical diagonalization does not match reference energy.")
    print(f"  Difference: {abs(E_classical_total - E_reference_total):.10e} Ha")
    print("Proceeding with calculation but results may be inconsistent.")

# =============================================================================
# QUANTUM MEASUREMENT SIMULATOR
# =============================================================================

class QuantumEstimator:
    """
    Quantum measurement simulator with configurable shot noise.
    
    Implements statevector-based expectation value calculation with
    optional finite sampling noise modeled using proper operator variance.
    
    Parameters:
        shots (int or None): Number of measurement shots. None for exact.
        operator (SparsePauliOp): Observable operator for variance calculation.
    
    Noise Model:
        For finite shots, measurement uncertainty is modeled as:
        sigma = sqrt(Var[H] / shots)
        where Var[H] = <psi|H^2|psi> - <psi|H|psi>^2
    """
    
    def __init__(self, shots=None, operator=None):
        self.shots = shots
        self.operator_squared = None
        self.call_count = 0
        
        # Pre-compute H^2 for variance calculations
        if shots is not None and operator is not None:
            self.operator_squared = operator @ operator
    
    def run(self, pubs, *, precision=None):
        """
        Execute measurement simulation on provided circuits.
        
        Args:
            pubs: List of tuples (circuit, observable, parameters) or PUB objects
            precision: Optional precision parameter (unused in exact simulation)
        """
        # Handle both old and new API formats
        results = []
        
        # Parse input format
        if isinstance(pubs, list) and len(pubs) > 0:
            first_pub = pubs[0]
            
            # New API: list of tuples (circuit, observable, parameters)
            if isinstance(first_pub, tuple) and len(first_pub) == 3:
                for circuit, observable, params in pubs:
                    self.call_count += 1
                    
                    # Bind parameters and compute statevector
                    if params is not None and len(params) > 0:
                        # Handle parameter binding - params might be nested or flat
                        # Flatten if needed (handles both 1D array and nested structure)
                        if hasattr(params, 'flatten'):
                            param_values = params.flatten()
                        else:
                            param_values = params

                        # Create parameter dictionary for binding
                        param_dict = dict(zip(circuit.parameters, param_values))
                        bound_circuit = circuit.assign_parameters(param_dict)
                    else:
                        bound_circuit = circuit
                    
                    statevector = Statevector(bound_circuit)
                    
                    # Calculate exact expectation value
                    expectation_value = statevector.expectation_value(observable).real
                    
                    # Apply shot noise if configured
                    if self.shots is not None and self.operator_squared is not None:
                        # Calculate operator variance
                        expectation_H2 = statevector.expectation_value(self.operator_squared).real
                        variance = expectation_H2 - expectation_value**2
                        
                        # Add Gaussian noise with proper standard error
                        if variance > 0:
                            standard_error = np.sqrt(variance / self.shots)
                            noise_sample = np.random.normal(0, standard_error)
                            expectation_value += noise_sample
                    
                    results.append(expectation_value)

        # Return a job-like object with a result() method
        # VQE expects job.result()[0] to get the first PubResult
        class FakeJob:
            def __init__(self, expectation_values):
                self._values = expectation_values

            def result(self):
                # Return list of PubResult objects
                pub_results = []
                for val in self._values:
                    pub_result = type('PubResult', (), {
                        'data': type('Data', (), {'evs': val})(),
                        'metadata': {}
                    })()
                    pub_results.append(pub_result)
                return pub_results

        return FakeJob(results)

# =============================================================================
# QUANTUM CIRCUIT CONSTRUCTION
# =============================================================================
print("\n" + "-" * 80)
print("Constructing Variational Ansatz")
print("-" * 80)

# Prepare initial state
if USE_HF_INIT:
    # Hartree-Fock reference: |1100> for H2
    # Represents two electrons in lowest spatial orbital
    initial_state = QuantumCircuit(H_operator.num_qubits)
    initial_state.x([0, 1])
    print("Initial state:         Hartree-Fock |1100>")
else:
    # Vacuum state |0000>
    initial_state = None
    print("Initial state:         Vacuum |0000>")

# Construct parameterized ansatz
ansatz = TwoLocal(
    num_qubits=H_operator.num_qubits,
    rotation_blocks=['ry', 'rz'],
    entanglement_blocks='cx',
    entanglement=ENTANGLEMENT,
    reps=ANSATZ_DEPTH,
    initial_state=initial_state
)

# Circuit statistics
num_parameters = ansatz.num_parameters
circuit_depth = ansatz.decompose().depth()
gate_operations = ansatz.decompose().count_ops()
total_gates = sum(gate_operations.values())

print(f"Ansatz parameters:     {num_parameters}")
print(f"Circuit depth:         {circuit_depth}")
print(f"Total gate count:      {total_gates}")
print(f"Gate distribution:     {dict(gate_operations)}")

# =============================================================================
# OPTIMIZATION ALGORITHM CONFIGURATION
# =============================================================================
print("\n" + "-" * 80)
print("Configuring Classical Optimizer")
print("-" * 80)

optimizer_instances = {
    'COBYLA': COBYLA(maxiter=MAX_ITERATIONS, tol=1e-6),
    'SPSA': SPSA(maxiter=MAX_ITERATIONS),
    'SLSQP': SLSQP(maxiter=MAX_ITERATIONS, tol=1e-6)
}

if OPTIMIZER not in optimizer_instances:
    print(f"ERROR: Unknown optimizer '{OPTIMIZER}'")
    print(f"Available options: {list(optimizer_instances.keys())}")
    exit(1)

optimizer = optimizer_instances[OPTIMIZER]
print(f"Selected optimizer:    {OPTIMIZER}")
print(f"Maximum iterations:    {MAX_ITERATIONS}")

# =============================================================================
# VQE EXECUTION ENGINE
# =============================================================================

def execute_vqe_trial(trial_number=None):
    """
    Execute a single VQE optimization trial.
    
    Returns:
        history (dict): Convergence data for all iterations
        result: VQE result object containing optimal parameters and energy
        evaluations (int): Total number of cost function evaluations
    """
    
    # Initialize tracking structures
    convergence_history = {
        'iteration': [],
        'electronic_energy': [],
        'total_energy': [],
        'energy_error': [],
        'parameters': []
    }
    
    def iteration_callback(iteration_count, parameters, energy_mean, energy_std):
        """Callback function executed after each optimization iteration."""
        
        # Calculate derived quantities
        electronic_energy = energy_mean
        total_energy = energy_mean + E_nuclear
        energy_error = total_energy - E_reference_total
        
        # Record iteration data
        convergence_history['iteration'].append(iteration_count)
        convergence_history['electronic_energy'].append(electronic_energy)
        convergence_history['total_energy'].append(total_energy)
        convergence_history['energy_error'].append(energy_error)
        convergence_history['parameters'].append(parameters.copy())
        
        # Console output at specified intervals
        if VERBOSE and (iteration_count % 20 == 0 or iteration_count == 1):
            trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""
            error_kcal = energy_error * 627.5  # Convert to kcal/mol
            
            print(f"{trial_prefix}Iteration {iteration_count:3d}: "
                  f"E_total = {total_energy:.10f} Ha, "
                  f"Error = {energy_error:+.8f} Ha "
                  f"({error_kcal:+.3f} kcal/mol)")
    
    # Initialize quantum estimator
    estimator = QuantumEstimator(shots=SHOTS, operator=H_operator)
    
    # Configure and execute VQE
    vqe_instance = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=iteration_callback
    )
    
    vqe_result = vqe_instance.compute_minimum_eigenvalue(H_operator)
    
    return convergence_history, vqe_result, estimator.call_count

# =============================================================================
# EXPERIMENTAL EXECUTION
# =============================================================================
print("\n" + "=" * 80)
print("EXECUTING VQE OPTIMIZATION")
print("=" * 80)

all_convergence_histories = []
all_vqe_results = []
execution_start_time = datetime.now()

# Determine number of trials to execute
num_trials_to_run = NUM_TRIALS if (RUN_MULTIPLE_TRIALS and SHOTS is not None) else 1

if num_trials_to_run > 1:
    print(f"\nExecuting {num_trials_to_run} independent trials for statistical analysis\n")

for trial_index in range(num_trials_to_run):
    if num_trials_to_run > 1:
        print(f"Trial {trial_index + 1}/{num_trials_to_run}")
    
    history, result, num_evaluations = execute_vqe_trial(
        trial_number=(trial_index + 1) if num_trials_to_run > 1 else None
    )
    
    all_convergence_histories.append(history)
    all_vqe_results.append(result)
    
    if num_trials_to_run > 1:
        final_total_energy = result.eigenvalue.real + E_nuclear
        print(f"  Final energy: {final_total_energy:.10f} Ha\n")

execution_duration = (datetime.now() - execution_start_time).total_seconds()

print(f"\nOptimization completed")
print(f"Total execution time:  {execution_duration:.2f} seconds")
print(f"Function evaluations:  {num_evaluations}")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

# Primary results from first trial
primary_history = all_convergence_histories[0]
primary_result = all_vqe_results[0]

# Calculate final energies
E_vqe_electronic = primary_result.eigenvalue.real
E_vqe_total = E_vqe_electronic + E_nuclear

# Error metrics
absolute_error = E_vqe_total - E_reference_total
absolute_error_kcal = absolute_error * 627.5
relative_error = abs(absolute_error / E_reference_total) * 100

print("\n" + "-" * 80)
print("Final Energy Results")
print("-" * 80)
print(f"{'Quantity':<30} {'VQE':<20} {'Reference':<20} {'Difference':<15}")
print("-" * 80)
print(f"{'Electronic energy (Ha)':<30} {E_vqe_electronic:<20.10f} "
      f"{E_reference_electronic:<20.10f} {E_vqe_electronic - E_reference_electronic:<+15.10f}")
print(f"{'Nuclear repulsion (Ha)':<30} {E_nuclear:<20.10f} "
      f"{E_nuclear:<20.10f} {'---':<15}")
print(f"{'Total energy (Ha)':<30} {E_vqe_total:<20.10f} "
      f"{E_reference_total:<20.10f} {absolute_error:<+15.10f}")

print("\n" + "-" * 80)
print("Error Metrics")
print("-" * 80)
print(f"Absolute error:        {abs(absolute_error):.10f} Ha")
print(f"Error (kcal/mol):      {abs(absolute_error_kcal):.4f} kcal/mol")
print(f"Relative error:        {relative_error:.6f} %")

# Chemical accuracy assessment (1 kcal/mol = 0.0016 Ha)
CHEMICAL_ACCURACY_THRESHOLD = 0.0016
chemical_accuracy_achieved = abs(absolute_error) < CHEMICAL_ACCURACY_THRESHOLD

print(f"\nChemical accuracy (< 1 kcal/mol): {chemical_accuracy_achieved}")
if not chemical_accuracy_achieved:
    print(f"  Current error:       {abs(absolute_error):.6f} Ha")
    print(f"  Required:            < {CHEMICAL_ACCURACY_THRESHOLD:.6f} Ha")
    print(f"  Deficit:             {abs(absolute_error) - CHEMICAL_ACCURACY_THRESHOLD:.6f} Ha")

# Multi-trial statistics
if len(all_vqe_results) > 1:
    print("\n" + "-" * 80)
    print("Multi-Trial Statistical Analysis")
    print("-" * 80)
    
    final_energies = [r.eigenvalue.real + E_nuclear for r in all_vqe_results]
    energy_mean = np.mean(final_energies)
    energy_std = np.std(final_energies, ddof=1)
    energy_min = np.min(final_energies)
    energy_max = np.max(final_energies)
    energy_range = energy_max - energy_min
    
    print(f"Sample size:           {len(final_energies)}")
    print(f"Mean energy:           {energy_mean:.10f} Ha")
    print(f"Standard deviation:    {energy_std:.10f} Ha")
    print(f"Minimum energy:        {energy_min:.10f} Ha")
    print(f"Maximum energy:        {energy_max:.10f} Ha")
    print(f"Range:                 {energy_range:.10f} Ha")
    print(f"Std error of mean:     {energy_std / np.sqrt(len(final_energies)):.10f} Ha")

# Convergence diagnostics
if len(primary_history['energy_error']) >= 20:
    print("\n" + "-" * 80)
    print("Convergence Diagnostics")
    print("-" * 80)
    
    # Analyze final iterations
    final_window = 20
    final_errors = primary_history['energy_error'][-final_window:]
    
    error_mean_final = np.mean([abs(e) for e in final_errors])
    error_std_final = np.std(final_errors, ddof=1)
    error_max_deviation = np.max(np.abs(final_errors))
    
    print(f"Analysis window:       Final {final_window} iterations")
    print(f"Mean |error|:          {error_mean_final:.10f} Ha")
    print(f"Standard deviation:    {error_std_final:.10f} Ha")
    print(f"Maximum deviation:     {error_max_deviation:.10f} Ha")
    
    # Convergence stability assessment
    if error_std_final < 1e-4:
        stability_assessment = "High stability"
    elif error_std_final < 1e-3:
        stability_assessment = "Moderate stability"
    else:
        stability_assessment = "Low stability (consider increasing iterations)"
    
    print(f"Stability assessment:  {stability_assessment}")

# =============================================================================
# DATA EXPORT
# =============================================================================
print("\n" + "-" * 80)
print("Exporting Data")
print("-" * 80)

# Create output directory
output_directory = Path("vqe_results")
output_directory.mkdir(exist_ok=True)

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Export primary trial data to CSV
csv_filename = output_directory / f"vqe_convergence_d{ANSATZ_DEPTH}_{OPTIMIZER}_{timestamp}.csv"

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write metadata header
    csv_writer.writerow(["# VQE Convergence Data"])
    csv_writer.writerow(["# System:", data.get('molecule', 'H2')])
    csv_writer.writerow(["# Ansatz depth:", ANSATZ_DEPTH])
    csv_writer.writerow(["# Optimizer:", OPTIMIZER])
    csv_writer.writerow(["# Shots:", SHOTS if SHOTS else "Exact"])
    csv_writer.writerow(["# Final error (Ha):", f"{absolute_error:.10f}"])
    csv_writer.writerow(["# Chemical accuracy achieved:", chemical_accuracy_achieved])
    csv_writer.writerow([])
    
    # Write column headers
    column_headers = [
        "Iteration",
        "Total_Energy_Ha",
        "Electronic_Energy_Ha",
        "Error_Ha",
        "Error_kcal_mol"
    ] + [f"Parameter_{i}" for i in range(num_parameters)]
    csv_writer.writerow(column_headers)
    
    # Write iteration data
    for i in range(len(primary_history['iteration'])):
        row_data = [
            primary_history['iteration'][i],
            primary_history['total_energy'][i],
            primary_history['electronic_energy'][i],
            primary_history['energy_error'][i],
            primary_history['energy_error'][i] * 627.5
        ] + list(primary_history['parameters'][i])
        csv_writer.writerow(row_data)

print(f"Convergence data:      {csv_filename}")

# Export summary report
summary_filename = output_directory / f"vqe_summary_{timestamp}.txt"

with open(summary_filename, 'w') as summary_file:
    summary_file.write("VQE EXPERIMENTAL REPORT\n")
    summary_file.write("=" * 80 + "\n\n")
    
    summary_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    summary_file.write("SYSTEM CONFIGURATION\n")
    summary_file.write("-" * 80 + "\n")
    summary_file.write(f"Molecular system:      {data.get('molecule', 'H2')}\n")
    summary_file.write(f"Bond length:           {data.get('bond_length', data.get('bond_length_angstrom', 'N/A'))} Angstrom\n")
    summary_file.write(f"Nuclear repulsion:     {E_nuclear:.10f} Ha\n")
    summary_file.write(f"Reference energy:      {E_reference_total:.10f} Ha\n\n")
    
    summary_file.write("COMPUTATIONAL PARAMETERS\n")
    summary_file.write("-" * 80 + "\n")
    summary_file.write(f"Ansatz depth:          {ANSATZ_DEPTH}\n")
    summary_file.write(f"Ansatz parameters:     {num_parameters}\n")
    summary_file.write(f"Circuit depth:         {circuit_depth}\n")
    summary_file.write(f"Entanglement:          {ENTANGLEMENT}\n")
    summary_file.write(f"Initial state:         {'Hartree-Fock' if USE_HF_INIT else 'Vacuum'}\n")
    summary_file.write(f"Optimizer:             {OPTIMIZER}\n")
    summary_file.write(f"Max iterations:        {MAX_ITERATIONS}\n")
    summary_file.write(f"Shot noise:            {SHOTS if SHOTS else 'None'}\n\n")
    
    summary_file.write("RESULTS\n")
    summary_file.write("-" * 80 + "\n")
    summary_file.write(f"VQE total energy:      {E_vqe_total:.10f} Ha\n")
    summary_file.write(f"Reference energy:      {E_reference_total:.10f} Ha\n")
    summary_file.write(f"Absolute error:        {absolute_error:+.10f} Ha\n")
    summary_file.write(f"Error (kcal/mol):      {absolute_error_kcal:+.4f} kcal/mol\n")
    summary_file.write(f"Relative error:        {relative_error:.6f} %\n")
    summary_file.write(f"Chemical accuracy:     {chemical_accuracy_achieved}\n\n")
    
    summary_file.write("PERFORMANCE\n")
    summary_file.write("-" * 80 + "\n")
    summary_file.write(f"Total iterations:      {len(primary_history['iteration'])}\n")
    summary_file.write(f"Function evaluations:  {num_evaluations}\n")
    summary_file.write(f"Execution time:        {execution_duration:.2f} seconds\n")

print(f"Summary report:        {summary_filename}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "-" * 80)
print("Generating Figures")
print("-" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

# Color scheme
color_vqe = '#1f77b4'
color_reference = '#d62728'
color_chem_accuracy = '#2ca02c'

# Subplot 1: Energy convergence
ax1 = fig.add_subplot(gs[0, :])

# Plot all trials if multiple exist
if len(all_convergence_histories) > 1:
    for idx, history in enumerate(all_convergence_histories):
        alpha_value = 0.3 if idx > 0 else 1.0
        label_text = 'VQE' if idx == 0 else None
        ax1.plot(history['iteration'], history['total_energy'],
                color=color_vqe, linewidth=2, alpha=alpha_value, label=label_text)
else:
    ax1.plot(primary_history['iteration'], primary_history['total_energy'],
            color=color_vqe, linewidth=2, label='VQE')

ax1.axhline(y=E_reference_total, color=color_reference, linestyle='--',
           linewidth=2, label='Reference FCI')
ax1.fill_between(primary_history['iteration'],
                E_reference_total - CHEMICAL_ACCURACY_THRESHOLD,
                E_reference_total + CHEMICAL_ACCURACY_THRESHOLD,
                color=color_chem_accuracy, alpha=0.15,
                label='Chemical accuracy region')

ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Total Energy (Hartree)', fontsize=11)
shot_label = f" (Shots={SHOTS})" if SHOTS else " (Exact)"
ax1.set_title(f'Energy Convergence: Depth={ANSATZ_DEPTH}, {OPTIMIZER}{shot_label}',
             fontsize=12)
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

# Subplot 2: Parameter evolution
ax2 = fig.add_subplot(gs[1, :])
parameters_array = np.array(primary_history['parameters'])
num_params_to_plot = min(6, parameters_array.shape[1])

for param_idx in range(num_params_to_plot):
    ax2.plot(primary_history['iteration'], parameters_array[:, param_idx],
            label=f'θ_{param_idx}', linewidth=1.5, alpha=0.8)

ax2.axhline(y=0, color='black', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Parameter Value (radians)', fontsize=11)
ax2.set_title('Parameter Trajectory Evolution', fontsize=12)
ax2.legend(fontsize=9, ncol=2, loc='right')
ax2.grid(True, alpha=0.3)

# Subplot 3: Error magnitude (log scale)
ax3 = fig.add_subplot(gs[2, 0])
absolute_errors = [abs(e) for e in primary_history['energy_error']]
ax3.semilogy(primary_history['iteration'], absolute_errors,
            color='#ff7f0e', linewidth=2)
ax3.axhline(y=CHEMICAL_ACCURACY_THRESHOLD, color=color_chem_accuracy,
           linestyle='--', linewidth=2, label='Chemical accuracy')
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('|Error| (Hartree, log scale)', fontsize=11)
ax3.set_title('Error Magnitude Convergence', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, which='both', alpha=0.3)

# Subplot 4: Error in kcal/mol
ax4 = fig.add_subplot(gs[2, 1])
errors_kcal = [e * 627.5 for e in primary_history['energy_error']]
ax4.plot(primary_history['iteration'], errors_kcal,
        color='#9467bd', linewidth=2)
ax4.axhline(y=1, color=color_chem_accuracy, linestyle='--', linewidth=2)
ax4.axhline(y=-1, color=color_chem_accuracy, linestyle='--', linewidth=2)
ax4.axhline(y=0, color='black', linewidth=0.5, linestyle=':', alpha=0.5)
ax4.fill_between(primary_history['iteration'], -1, 1,
                color=color_chem_accuracy, alpha=0.15)
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Error (kcal/mol)', fontsize=11)
ax4.set_title('Error (Chemical Units)', fontsize=12)
ax4.grid(True, alpha=0.3)

# Subplot 5: Energy distribution
ax5 = fig.add_subplot(gs[3, 0])

if len(all_vqe_results) > 1:
    # Multi-trial: distribution of final energies
    final_energies = [r.eigenvalue.real + E_nuclear for r in all_vqe_results]
    ax5.hist(final_energies, bins=20, color=color_vqe, alpha=0.7, edgecolor='black')
    ax5.axvline(x=E_reference_total, color=color_reference, linestyle='--',
               linewidth=2, label='Reference')
    ax5.axvline(x=np.mean(final_energies), color='#17becf', linestyle='-',
               linewidth=2, label='Mean')
    ax5.set_title(f'Final Energy Distribution ({NUM_TRIALS} trials)', fontsize=12)
else:
    # Single trial: distribution of final iterations
    final_window_size = min(50, len(primary_history['total_energy']))
    final_energies = primary_history['total_energy'][-final_window_size:]
    ax5.hist(final_energies, bins=20, color=color_vqe, alpha=0.7, edgecolor='black')
    ax5.axvline(x=E_reference_total, color=color_reference, linestyle='--',
               linewidth=2, label='Reference')
    ax5.axvline(x=np.mean(final_energies), color='#17becf', linestyle='-',
               linewidth=2, label='Mean')
    ax5.set_title(f'Energy Distribution (Final {final_window_size} Iterations)', fontsize=12)

ax5.set_xlabel('Energy (Hartree)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.legend(fontsize=10)

# Subplot 6: Summary panel
ax6 = fig.add_subplot(gs[3, 1])
ax6.axis('off')

# Accuracy status
if chemical_accuracy_achieved:
    accuracy_status = "Achieved"
    recommendation = f"Consider reducing depth to {max(1, ANSATZ_DEPTH-1)}"
elif abs(absolute_error) < 0.01:
    accuracy_status = "Not achieved (within 10x threshold)"
    recommendation = f"Increase depth to {ANSATZ_DEPTH + 1}"
else:
    accuracy_status = "Not achieved (>10x threshold)"
    recommendation = f"Increase depth to {ANSATZ_DEPTH + 2}"

summary_text = f"""
EXPERIMENTAL SUMMARY
{'─' * 44}

Configuration:
  Ansatz depth:      {ANSATZ_DEPTH}
  Parameters:        {num_parameters}
  Circuit depth:     {circuit_depth}
  Optimizer:         {OPTIMIZER}
  Iterations:        {len(primary_history['iteration'])}
  Shot noise:        {f'{SHOTS} shots' if SHOTS else 'Disabled'}
  Initial state:     {'HF' if USE_HF_INIT else 'Vacuum'}

Results:
  VQE energy:        {E_vqe_total:.10f} Ha
  Reference:         {E_reference_total:.10f} Ha
  Error:             {absolute_error:+.10f} Ha
                     ({absolute_error_kcal:+.3f} kcal/mol)

Analysis:
  Chemical accuracy: {accuracy_status}
  Execution time:    {execution_duration:.1f} seconds
  
{'─' * 44}
Recommendation:
  {recommendation}
"""

ax6.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

# Overall figure title
fig.suptitle(f'VQE Analysis: H2 Molecular Ground State',
            fontsize=14, y=0.995)

# Save figure
if SAVE_PLOTS:
    figure_filename = output_directory / f"vqe_analysis_d{ANSATZ_DEPTH}_{OPTIMIZER}_{timestamp}.png"
    plt.savefig(figure_filename, dpi=150, bbox_inches='tight')
    print(f"Analysis figure:       {figure_filename}")

plt.show()

# =============================================================================
# EXPERIMENTAL CONCLUSIONS
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENTAL ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput files generated:")
print(f"  Data export:         {csv_filename.name}")
print(f"  Summary report:      {summary_filename.name}")
if SAVE_PLOTS:
    print(f"  Analysis figure:     {figure_filename.name}")

print(f"\nAll files located in: {output_directory}/")

print("\n" + "=" * 80)
print("Recommendations for further investigation:")
print("-" * 80)

if chemical_accuracy_achieved:
    print(f"  1. Reduce ansatz depth to {max(1, ANSATZ_DEPTH-1)} to test computational efficiency")
    print(f"  2. Enable shot noise (SHOTS = 4096) to assess robustness")
    print(f"  3. Execute multiple trials (RUN_MULTIPLE_TRIALS = True) for error bars")
    
elif abs(absolute_error) < 0.01:
    print(f"  1. Increase ansatz depth to {ANSATZ_DEPTH + 1}")
    print(f"  2. Increase maximum iterations to {MAX_ITERATIONS + 100}")
    if not USE_HF_INIT:
        print(f"  3. Enable Hartree-Fock initialization (USE_HF_INIT = True)")
    if ENTANGLEMENT == 'linear':
        print(f"  3. Consider 'full' entanglement topology for enhanced expressibility")
    
else:
    print(f"  1. Increase ansatz depth to {ANSATZ_DEPTH + 2}")
    print(f"  2. Enable Hartree-Fock initialization (USE_HF_INIT = True)")
    print(f"  3. Increase maximum iterations to {MAX_ITERATIONS + 200}")
    if OPTIMIZER != 'COBYLA':
        print(f"  4. Test COBYLA optimizer for improved robustness")

print("\n" + "=" * 80)