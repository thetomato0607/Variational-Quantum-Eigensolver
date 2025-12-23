#!/bin/bash

echo "Initializing VQE Research Project Structure..."

# 1. Create Root Level Files
touch README.md LICENSE pyproject.toml .gitignore .pre-commit-config.yaml
touch requirements.txt

# 2. Create Directory Skeleton
mkdir -p .github/workflows
mkdir -p src/vqe/hamiltonians
mkdir -p src/vqe/ansatz
mkdir -p src/vqe/backends
mkdir -p src/vqe/optimizers
mkdir -p experiments
mkdir -p scripts
mkdir -p results/tfim/figures
mkdir -p results/h2/figures
mkdir -p report/figures
mkdir -p tests
mkdir -p data/raw      # Added based on recommendation
mkdir -p data/processed

# 3. Create Source Code Files (__init__.py makes them importable packages)
touch src/vqe/__init__.py
touch src/vqe/config.py
touch src/vqe/measurement.py
touch src/vqe/vqe_runner.py
touch src/vqe/metrics.py
touch src/vqe/plotting.py
touch src/vqe/utils.py

# Sub-modules
touch src/vqe/hamiltonians/__init__.py
touch src/vqe/hamiltonians/tfim.py
touch src/vqe/hamiltonians/h2.py

touch src/vqe/ansatz/__init__.py
touch src/vqe/ansatz/hardware_efficient.py
touch src/vqe/ansatz/ucc_like.py

touch src/vqe/backends/__init__.py
touch src/vqe/backends/ideal.py
touch src/vqe/backends/shot_based.py
touch src/vqe/backends/noisy.py

touch src/vqe/optimizers/__init__.py
touch src/vqe/optimizers/scipy_opt.py
touch src/vqe/optimizers/spsa.py

# 4. Create Placeholder Notebooks & Scripts
touch experiments/00_quickstart.ipynb
touch experiments/01_tfim_ideal_vs_shots.ipynb
touch experiments/02_optimizer_comparison.ipynb
touch experiments/03_ansatz_comparison.ipynb
touch experiments/04_noisy_runs.ipynb

touch scripts/run_tfim_grid.py
touch scripts/run_noisy.py

# 5. Create Report & Tests
touch report/vqe_report.md
touch tests/test_hamiltonians.py
touch tests/test_ansatz.py
touch tests/test_expectation.py
touch tests/test_vqe_smoke.py

# 6. Add basic .gitignore content
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo "vqe_env/" >> .gitignore
echo ".vscode/" >> .gitignore
echo ".DS_Store" >> .gitignore

echo "Project structure created successfully!"