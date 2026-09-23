# Benchmarking VQE Strategies for Molecular Hydrogen on NISQ Architectures
Author: Frankie Lam
Affiliation: Department of Physics, University College London
Contact: yuilonlam0607@gmail.com

## 1. Introduction

This repository contains the source code and experimental data for benchmarking the Variational Quantum Eigensolver (VQE) on Noisy Intermediate-Scale Quantum (NISQ) architectures. The project investigates the critical trade-offs between ansatz expressibility (UCCSD vs. TwoLocal) and measurement overhead for the Hydrogen molecule ($H_2$) across its dissociation curve ($0.5\text{\AA}$ to $2.5\text{\AA}$). The study explicitly isolates failure modes in the bond-dissociation regime ($R=1.5\text{\AA}$) and quantifies the "noise floor" using a local Aer depolarizing-noise simulator standing in for hardware noise.
   

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

4. Simulated Hardware Noise
Executed on a local Qiskit Aer simulator using a depolarizing-noise model (`src/vqe/backends/noisy.py`), not on real IBM Quantum hardware.
- Noise Floor: A systematic energy error remains even with optimal stochastic strategies under this noise model, representing the limit of unmitigated noise in the simulated setting.


## 3. Repository Structure
```text
Variational-Quantum-Eigensolver/
├── src/vqe/               # importable package
│   ├── hamiltonians/      # H2 (PySCF, STO-3G) and TFIM qubit Hamiltonians
│   ├── ansatz/            # TwoLocal (hardware-efficient) and UCCSD
│   ├── backends/          # ideal, shot-based and noisy (Aer) estimators
│   ├── optimizers/        # COBYLA and SPSA factories
│   ├── vqe_runner.py      # VQE wrapper that records the energy history
│   ├── metrics.py, plotting.py, measurement.py, utils.py, config.py
├── scripts/               # one script per figure; run from the repo root
├── results/h2/figures/    # H2 figures
├── results/tfim/figures/  # TFIM figure
├── tests/                 # pytest suite
├── .github/workflows/     # CI: tests + lint
├── pyproject.toml         # package metadata and pinned dependencies
└── requirements.txt       # the same pins, for pip install -r
```

## 4. Installation & Dependencies
Requires Python 3.11 (the version CI tests) on Linux or macOS; PySCF has no Windows wheels, so use WSL on Windows. All dependencies are pinned in `pyproject.toml`.

```bash
git clone https://github.com/thetomato0607/Variational-Quantum-Eigensolver.git
cd Variational-Quantum-Eigensolver
python -m venv .venv
source .venv/bin/activate        # Windows (WSL): same command
pip install -e .                 # installs the vqe package and pinned dependencies
pip install pytest && pytest tests/
```

## 5. Reproducibility
Run every script from the repository root; each writes its figure to the path shown.

| Figure / number in section 2 | Command | Status |
|---|---|---|
| `results/tfim/figures/tfim_scan.png` | `python scripts/run_tfim_grid.py` | Runs |
| `results/h2/figures/ansatz_comparison.png`; TwoLocal error at 1.5 Å, iteration counts | `python scripts/run_ansatz_comparison.py` | Runs; the TwoLocal error varies between runs because the initial point is unseeded (one re-run gave ~90 mHa) |
| `results/h2/figures/dissociation_curve.png` | `python scripts/run_h2_scan.py` | **Currently fails** with the pinned versions: it passes a V2 estimator to qiskit-algorithms' VQE |
| `results/h2/figures/noise_comparison.png` | `python scripts/run_noise_comparison.py` | **Currently fails** for the same reason (Aer `EstimatorV2`) |
| `results/h2/figures/accuracy_benchmark.png`; SPSA −1.1167 Ha | `python scripts/plot_benchmark.py` | Plots hard-coded energies from an earlier noisy-simulator run that is not recorded here |
| UCCSD chemical accuracy at equilibrium | — | No script in this repository runs UCCSD at 0.735 Å |

Only the noisy-simulation backend is seeded: `src/vqe/backends/noisy.py` sets `estimator.options.seed_simulator = 42` on the Aer noise sampler. Ansatz parameter initialization and SPSA's perturbation sampling (`src/vqe/optimizers/spsa.py`) do not take a seed, so runs that exercise those paths are not fully deterministic.

## 6. Acknowledgements
- University College London (UCL) Department of Physics.
- AI tools were used for parts of the documentation and code; section 8 lists every AI-assisted change.

## 7. License
This project is licensed under the MIT License - see the LICENSE file for details.

## 8. AI assistance

Parts of this repository were written or changed with AI assistants. Affected code is marked in place with comments of the form `AI-assisted (<tool>, <commit>)`; list them with `git grep -n "AI-assisted"`.

- `2902ae0` (OpenAI Codex, merged via PR #1): `VQERunner` logging controls (`verbose`, `print_every`) and type hints.
- `da5aa23` (Claude): `pyproject.toml`, `requirements.txt` pins, `.pre-commit-config.yaml`, README accuracy fixes, removal of unrelated `vqc/` and `archive_offline/` folders.
- `ccab99a` (Claude): `.github/workflows/tests.yml`.
- `6645992` (Claude): `src/vqe/backends/ideal.py` switched to the V1 `Estimator` to fix a VQE crash.
- `531911f` (Claude): docstrings and explanatory comments across the code.
- The commit after `531911f` (Claude): README corrections, removal of the one-off `setup_project.sh` and `generate_tree.py` helpers, and relabelling of the accuracy benchmark from "Cloud"/"Hardware Validation" to simulator.
