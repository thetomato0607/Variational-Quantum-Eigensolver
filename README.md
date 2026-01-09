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
```text
Variational-Quantum-Eigensolver/
├── src/
│   ├── vqe/
│   │   ├── measurement.py
│   │   ├── plotting.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   ├── utils.py
│   │   ├── vqe_runner.py
│   │   ├── hamiltonians/
│   │   │   ├── tfim.py
│   │   │   ├── h2.py
│   │   ├── ansatz/
│   │   │   ├── ucc_like.py
│   │   │   ├── hardware_efficient.py
│   │   ├── optimizers/
│   │   │   ├── spsa.py
│   │   │   ├── scipy_opt.py
│   │   ├── backends/
│   │   │   ├── shot_based.py
│   │   │   ├── noisy.py
│   │   │   ├── ideal.py
├── scripts/
│   ├── run_h2_scan.py
│   ├── run_tfim_grid.py
│   ├── run_ansatz_comparison.py
│   ├── run_noise_comparison.py
│   ├── plot_benchmark.py
├── results/
│   ├── h2/
│   │   ├── figures/
│   │   │   ├── accuracy_benchmark.png
│   │   │   ├── noise_comparison.png
│   │   │   ├── dissociation_curve.png
│   │   │   ├── ansatz_comparison.png
│   ├── tfim/
│       ├── figures/
│           ├── tfim_scan.png
├── tests/
│   ├── test_vqe_smoke.py
│   ├── test_hamiltonians.py
│   ├── test_expectation.py
│   ├── test_ansatz.py
├── requirements.txt
├── README.md
```

## 4. Installation & Dependencies
To ensure scientific reproducibility, this project uses specific versions of Qiskit and PySCF.

Clone the repository:
git clone https://github.com/thetomato0607/Variational-Quantum-Eigensolver.git
cd Variational-Quantum-Eigensolver


Create a virtual environment (Recommended):
python -m venv vqe_env
source vqe_env/bin/activate  # On Windows: vqe_env\Scripts\activate

Install dependencies:
pip install -r requirements.txt


Key Dependencies:
- qiskit>=1.0
- pyscf>=2.5
- qiskit-aer
- qiskit-nature
- matplotlib
- numpy

## 5. Reproducibility
All stochastic processes (initialization, measurement sampling, optimizer perturbation) are controlled via fixed random seeds (Seed: 1234) to ensure deterministic trajectories.

## 6. Acknowledge
- University College London (UCL) Department of Physics.
- IBM Quantum for providing access to cloud-based runtime primitives.
- Preliminary drafts of the documentation were edited for clarity using LLM tools; all data and analysis are original work.

## 7. Licensce
This project is licensed under the MIT License - see the LICENSE file for details.