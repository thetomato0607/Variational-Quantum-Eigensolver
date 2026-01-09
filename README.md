# Benchmarking VQE Strategies for Molecular Hydrogen on NISQ Architectures
Author: Frankie Lam
Affiliation: Department of Physics, University College London
Contact: yuilonlam0607@gmail.com

## 1. Introduction

This repository contains the source code and experimental data for benchmarking the Variational Quantum Eigensolver (VQE) on Noisy Intermediate-Scale Quantum (NISQ) architectures. The project investigates the critical trade-offs between ansatz expressibility (UCCSD vs. TwoLocal) and measurement overhead for the Hydrogen molecule ($H_2$) across its dissociation curve ($0.5\text{\AA}$ to $2.5\text{\AA}$)1.The study explicitly isolates failure modes in the bond-dissociation regime ($R=1.5\text{\AA}$) and quantifies the "noise floor" of current hardware using the IBM Quantum Runtime environment2.
   

## 2. Key Experiments & Findings

1. The Dissociation Comparison (UCCSD vs. TwoLocal)
We benchmarked the Chemically Inspired (UCCSD) ansatz against the Hardware-Efficient (TwoLocal) ansatz.
- Result: UCCSD achieves chemical accuracy ($<1.6$ mHa) at equilibrium.
- Failure Mode: TwoLocal fails significantly at $1.5\text{\AA}$ (Static Correlation regime) with an error exceeding 190 mHa, confirming that heuristic circuits lack the entanglement capacity for bond breaking.

2. Optimization Landscapes
- UCCSD: Convex-like landscape, converges in $<50$ iterations.
- TwoLocal: Rugged, non-convex landscape with high-amplitude oscillations, requiring $>175$ iterations to converge.

3. Optimizer Robustness (COBYLA vs. SPSA) 
Under simulated noise models ($\sigma=0.02$)
- COBYLA: Converges fast but traps in local minima due to noise sensitivity.
- SPSA: Exhibits high variance but successfully escapes local minima, achieving a lower final energy ($-1.1167$ Ha) than COBYLA.

4. Hardware Validation
Executed on IBM Quantum cloud backend.
- Noise Floor: A systematic error of ~20 mHa remains even with optimal stochastic strategies, representing the limit of unmitigated hardware.


## 3. Repository Structure
Variational-Quantum-Eigensolver
│   ├── setup_project.sh
│   ├── pyproject.toml
│   ├── generate_tree.py
│   ├── README.md
│   ├── .pre-commit-config.yaml
│   ├── requirements.txt
│   ├── results/
│   │   ├── h2/
│   │   │   ├── figures/
│   │   │   │   ├── accuracy_benchmark.png
│   │   │   │   ├── noise_comparison.png
│   │   │   │   ├── dissociation_curve.png
│   │   │   │   ├── ansatz_comparison.png
│   │   ├── tfim/
│   │   │   ├── figures/
│   │   │   │   ├── tfim_scan.png
│   ├── .github/
│   │   ├── workflows/
│   ├── src/
│   │   ├── vqe.egg-info/
│   │   │   ├── dependency_links.txt
│   │   │   ├── PKG-INFO
│   │   │   ├── SOURCES.txt
│   │   │   ├── top_level.txt
│   │   ├── vqe/
│   │   │   ├── measurement.py
│   │   │   ├── plotting.py
│   │   │   ├── config.py
│   │   │   ├── metrics.py
│   │   │   ├── utils.py
│   │   │   ├── vqe_runner.py
│   │   │   ├── __init__.py
│   │   │   ├── hamiltonians/
│   │   │   │   ├── tfim.py
│   │   │   │   ├── h2.py
│   │   │   │   ├── __init__.py
│   │   │   ├── ansatz/
│   │   │   │   ├── ucc_like.py
│   │   │   │   ├── hardware_efficient.py
│   │   │   │   ├── __init__.py
│   │   │   ├── optimizers/
│   │   │   │   ├── spsa.py
│   │   │   │   ├── scipy_opt.py
│   │   │   │   ├── __init__.py
│   │   │   ├── backends/
│   │   │   │   ├── shot_based.py
│   │   │   │   ├── noisy.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ideal.py
│   ├── scripts/
│   │   ├── run_h2_scan.py
│   │   ├── run_tfim_grid.py
│   │   ├── run_ansatz_comparison.py
│   │   ├── run_noise_comparison.py
│   │   ├── plot_benchmark.py
│   ├── .claude/
│   │   ├── settings.local.json
│   ├── archive_offline/
│   │   ├── flight_data.pkl
│   │   ├── pauli_terms.txt
│   │   ├── H2_data.py
│   │   ├── qiskit_tutorial.txt
│   │   ├── molecule_info.json
│   │   ├── hello_vqe.py
│   │   ├── vqe_lab.py
│   │   ├── pyscf_data.py
│   ├── vqe_env/
│   │   ├── pyvenv.cfg
│   │   ├── share/
│   │   │   ├── man/
│   │   │   │   ├── man1/
│   │   │   │   │   ├── ttx.1
│   ├── tests/
│   │   ├── test_vqe_smoke.py
│   │   ├── test_hamiltonians.py
│   │   ├── test_expectation.py
│   │   ├── test_ansatz.py

