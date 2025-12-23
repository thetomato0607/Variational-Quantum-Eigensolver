"""
VQE LIVE: REAL-TIME QUANTUM CHEMISTRY
=====================================
Uses PySCF to calculate H2 properties on the fly.
UPDATED FOR QISKIT 1.0+ / 2.0+
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np

import matplotlib

# Use a non-interactive backend when display isn't available.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA


@dataclass
class VQEResults:
    exact_total: float
    exact_electronic: float
    vqe_total: float
    vqe_electronic: float
    error_total: float
    error_kcal: float
    iterations: int
    optimal_params: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live VQE simulation for H2.")
    parser.add_argument("--bond-distance", type=float, default=0.735, help="H-H distance in Angstroms")
    parser.add_argument("--ansatz-depth", type=int, default=2, help="TwoLocal repetition depth")
    parser.add_argument("--max-iterations", type=int, default=100, help="Optimizer iterations")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for output artifacts")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    return parser.parse_args()


def get_estimator():
    """Return an estimator compatible with the installed Qiskit version."""
    try:
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
        estimator_name = "StatevectorEstimator"
    except ImportError:
        from qiskit.primitives import Estimator

        estimator = Estimator()
        estimator_name = "Estimator"

    return estimator, estimator_name


def build_problem(bond_distance: float):
    driver = PySCFDriver(
        atom=f"H 0.0 0.0 0.0; H 0.0 0.0 {bond_distance}",
        charge=0,
        spin=0,
        basis="sto-3g",
    )
    return driver.run()


def run_exact_solution(qubit_op):
    exact_solver = NumPyMinimumEigensolver()
    result_exact = exact_solver.compute_minimum_eigenvalue(qubit_op)
    return result_exact.eigenvalue.real


def run_vqe(qubit_op, ansatz, optimizer, estimator, e_nuclear, exact_total):
    convergence_history: list[tuple[int, float]] = []
    iteration_count = [0]

    def progress_callback(eval_count, parameters, energy_mean, energy_std):
        e_total = energy_mean + e_nuclear
        convergence_history.append((eval_count, e_total))
        iteration_count[0] = eval_count

        if eval_count % 10 == 0 or eval_count == 1:
            error = e_total - exact_total
            print(
                f"   Iteration {eval_count:3d}: E = {e_total:.8f} Ha, "
                f"Error = {error:+.6f} Ha"
            )

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=progress_callback,
    )

    result_vqe = vqe.compute_minimum_eigenvalue(qubit_op)

    return result_vqe, convergence_history, iteration_count[0]


def save_plot(output_dir: Path, bond_distance: float, ansatz_depth: int, exact_total: float,
              convergence_history: list[tuple[int, float]], chem_accuracy: float):
    iterations = [entry[0] for entry in convergence_history]
    energies = [entry[1] for entry in convergence_history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, "b-", linewidth=2, label="VQE Energy")
    plt.axhline(y=exact_total, color="r", linestyle="--", linewidth=2, label="Exact Energy")

    if iterations:
        plt.fill_between(
            iterations,
            exact_total - chem_accuracy,
            exact_total + chem_accuracy,
            color="green",
            alpha=0.2,
            label="Chemical Accuracy (±1 kcal/mol)",
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Total Energy (Hartree)", fontsize=12)
    plt.title(
        f"VQE Convergence: H₂ at {bond_distance} Å (Depth={ansatz_depth})",
        fontsize=14,
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    filename = output_dir / f"vqe_live_{bond_distance}A.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"   Plot saved: {filename}")
    return filename


def save_convergence_csv(output_dir: Path, bond_distance: float, exact_total: float,
                          convergence_history: list[tuple[int, float]]):
    csv_filename = output_dir / f"vqe_convergence_{bond_distance}A.csv"
    with csv_filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Energy_Ha", "Error_Ha", "Error_kcal_mol"])
        for iteration, energy in convergence_history:
            err = energy - exact_total
            writer.writerow([iteration, energy, err, err * 627.5])

    print(f"   Data saved: {csv_filename}")
    return csv_filename


def save_summary(output_dir: Path, bond_distance: float, ansatz_depth: int, max_iterations: int,
                 ansatz, results: VQEResults):
    summary_filename = output_dir / f"vqe_summary_{bond_distance}A.txt"
    with summary_filename.open("w") as f:
        f.write("VQE CALCULATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("System: H2 molecule\n")
        f.write(f"Bond distance: {bond_distance} Angstrom\n")
        f.write("Basis set: STO-3G\n\n")
        f.write("VQE Configuration:\n")
        f.write(f"  Ansatz depth:     {ansatz_depth}\n")
        f.write(f"  Parameters:       {ansatz.num_parameters}\n")
        f.write("  Optimizer:        COBYLA\n")
        f.write(f"  Max iterations:   {max_iterations}\n\n")
        f.write("Results:\n")
        f.write(f"  VQE energy:       {results.vqe_total:.10f} Ha\n")
        f.write(f"  Exact energy:     {results.exact_total:.10f} Ha\n")
        f.write(f"  Error:            {results.error_total:+.10f} Ha\n")
        f.write(f"  Error (kcal/mol): {results.error_kcal:+.4f} kcal/mol\n")
        f.write(f"  Iterations:       {results.iterations}\n")

    print(f"   Summary saved: {summary_filename}")
    return summary_filename


def main() -> int:
    args = parse_args()
    bond_distance = args.bond_distance
    ansatz_depth = args.ansatz_depth
    max_iterations = args.max_iterations
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VQE LIVE: CALCULATING H2 MOLECULE")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Bond distance:  {bond_distance} Angstrom")
    print(f"  Ansatz depth:   {ansatz_depth}")
    print(f"  Max iterations: {max_iterations}")

    print(f"\n1. Running PySCF for H2 at {bond_distance} Angstroms...")
    problem = build_problem(bond_distance)

    hamiltonian_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(hamiltonian_op)

    e_nuclear = problem.nuclear_repulsion_energy

    print("   PySCF calculation complete")
    print(f"   Number of qubits:     {qubit_op.num_qubits}")
    print(f"   Pauli terms:          {len(qubit_op)}")
    print(f"   Nuclear repulsion:    {e_nuclear:.8f} Ha")

    print("\n2. Calculating exact reference energy...")
    e_exact_electronic = run_exact_solution(qubit_op)
    e_exact_total = e_exact_electronic + e_nuclear

    print(f"   Exact electronic:     {e_exact_electronic:.10f} Ha")
    print(f"   Exact total:          {e_exact_total:.10f} Ha")

    print("\n3. Setting up VQE...")
    ansatz = TwoLocal(
        num_qubits=qubit_op.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=ansatz_depth,
    )

    print(f"   Ansatz parameters:    {ansatz.num_parameters}")
    print(f"   Circuit depth:        {ansatz.decompose().depth()}")

    optimizer = COBYLA(maxiter=max_iterations)
    estimator, estimator_name = get_estimator()
    print(f"   Using: {estimator_name}")

    print("\n4. Running VQE optimization...")
    print("   (This may take 1-3 minutes)\n")

    result_vqe, convergence_history, iteration_count = run_vqe(
        qubit_op,
        ansatz,
        optimizer,
        estimator,
        e_nuclear,
        e_exact_total,
    )

    e_vqe_electronic = result_vqe.eigenvalue.real
    e_vqe_total = e_vqe_electronic + e_nuclear
    error_total = e_vqe_total - e_exact_total
    error_kcal = error_total * 627.5

    results = VQEResults(
        exact_total=e_exact_total,
        exact_electronic=e_exact_electronic,
        vqe_total=e_vqe_total,
        vqe_electronic=e_vqe_electronic,
        error_total=error_total,
        error_kcal=error_kcal,
        iterations=iteration_count,
        optimal_params=len(result_vqe.optimal_point),
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nEnergies (Hartree):")
    print(f"  VQE total energy:     {results.vqe_total:.10f} Ha")
    print(f"  Exact total energy:   {results.exact_total:.10f} Ha")
    print(f"  Absolute error:       {abs(results.error_total):.10f} Ha")
    print(f"  Error (kcal/mol):     {abs(results.error_kcal):.4f} kcal/mol")

    print("\nPerformance:")
    print(f"  Total iterations:     {results.iterations}")
    print(f"  Optimal parameters:   {results.optimal_params}")

    chem_accuracy = 0.0016
    if abs(results.error_total) < chem_accuracy:
        print("\nChemical accuracy:    Achieved")
    else:
        print("\nChemical accuracy:    Not achieved")
        print(f"  Target:               < {chem_accuracy:.6f} Ha")
        print(f"  Current:              {abs(results.error_total):.6f} Ha")

    print("\n5. Generating convergence plot...")
    plot_file = save_plot(
        output_dir,
        bond_distance,
        ansatz_depth,
        results.exact_total,
        convergence_history,
        chem_accuracy,
    )

    print("\n6. Exporting data...")
    csv_file = save_convergence_csv(output_dir, bond_distance, results.exact_total, convergence_history)
    summary_file = save_summary(
        output_dir,
        bond_distance,
        ansatz_depth,
        max_iterations,
        ansatz,
        results,
    )

    print("\n" + "=" * 60)
    print("VQE CALCULATION COMPLETE")
    print("=" * 60)

    print("\nFiles generated:")
    print(f"  1. {plot_file}")
    print(f"  2. {csv_file}")
    print(f"  3. {summary_file}")

    print("\nFinal result:")
    print(f"  VQE:   {results.vqe_total:.8f} Ha")
    print(f"  Exact: {results.exact_total:.8f} Ha")
    print(f"  Error: {abs(results.error_total):.8f} Ha ({abs(results.error_kcal):.3f} kcal/mol)")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())