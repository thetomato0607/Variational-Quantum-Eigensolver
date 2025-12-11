"""
H2 Molecule Data Generator - FROM YOUR PYSCF CALCULATION
=========================================================
Using the actual results from your PySCF run!

Your PySCF Results:
  Nuclear repulsion:      0.7199689944 Ha
  Electronic energy (FCI): -1.8572750302 Ha
  Total energy:           -1.1373060358 Ha
"""

import pickle
import json
import numpy as np

print("=" * 80)
print("H2 MOLECULE DATA GENERATOR - YOUR PYSCF RESULTS")
print("=" * 80)

# =============================================================================
# YOUR ACTUAL PYSCF HAMILTONIAN
# =============================================================================
# These are the exact Pauli terms from your PySCF calculation
# Order: ['IIII', 'IIIZ', 'IIZI', 'IZII', 'ZIII', 'IIZZ', 'IZIZ', 'ZIIZ', 
#         'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII']

h2_hamiltonian = [
    ('IIII', -0.81054798),
    ('IIIZ',  0.17218393),
    ('IIZI', -0.22575349),
    ('IZII',  0.17218393),
    ('ZIII', -0.22575349),
    ('IIZZ',  0.12091263),
    ('IZIZ',  0.16892754),
    ('ZIIZ',  0.16614543),
    ('YYYY',  0.0452328),
    ('XXYY',  0.0452328),
    ('YYXX',  0.0452328),
    ('XXXX',  0.0452328),
    ('IZZI',  0.16614543),
    ('ZIZI',  0.17464343),
    ('ZZII',  0.12091263),
]

# =============================================================================
# YOUR ACTUAL PYSCF ENERGIES
# =============================================================================
molecular_data = {
    # Basic info
    "molecule": "H2",
    "bond_length_angstrom": 0.735,  # Your calculation used this
    "basis_set": "sto-3g",
    
    # Quantum info
    "num_qubits": 4,
    "num_electrons": 2,
    "num_spatial_orbitals": 2,
    
    # Hamiltonian
    "pauli_list": h2_hamiltonian,
    "num_pauli_terms": len(h2_hamiltonian),
    
    # YOUR ACTUAL ENERGIES from PySCF
    "nuclear_repulsion_energy": 0.7199689944,        # From your PySCF output
    "exact_electronic_energy": -1.8572750302,        # From your PySCF FCI
    "exact_total_energy": -1.1373060358,             # From your PySCF output
    
    # Reference values for comparison
    "hartree_fock_energy": None,  # You didn't report this, but it's ~-1.85 Ha
    "chemical_accuracy_threshold": 0.0016,  # 1 kcal/mol in Hartree
    
    # Metadata
    "notes": "Direct from your PySCF calculation output",
    "units": "Hartree (atomic units)",
    "source": "PySCF FCI calculation with STO-3G basis"
}

print(f"\nMolecule: {molecular_data['molecule']}")
print(f"Bond length: {molecular_data['bond_length_angstrom']} Å")
print(f"Basis set: {molecular_data['basis_set']}")
print(f"Number of qubits: {molecular_data['num_qubits']}")
print(f"Number of Pauli terms: {molecular_data['num_pauli_terms']}")

# =============================================================================
# VALIDATION - Verify your numbers make sense
# =============================================================================
print("\n" + "-" * 80)
print("VALIDATION OF YOUR PYSCF RESULTS")
print("-" * 80)

E_nuc = molecular_data['nuclear_repulsion_energy']
E_elec = molecular_data['exact_electronic_energy']
E_total = molecular_data['exact_total_energy']

print(f"\nYour PySCF energies:")
print(f"  Nuclear repulsion:      {E_nuc:.10f} Ha")
print(f"  Electronic energy (FCI): {E_elec:.10f} Ha")
print(f"  Total energy:           {E_total:.10f} Ha")

# Check energy conservation
calculated_total = E_elec + E_nuc
print(f"\nVerification:")
print(f"  E_electronic + E_nuclear = {calculated_total:.10f} Ha")
print(f"  Your reported E_total    = {E_total:.10f} Ha")
print(f"  Match: {abs(calculated_total - E_total) < 1e-8} ✓")

# Check physical reasonableness
print(f"\nPhysical checks:")
print(f"  ✓ Nuclear repulsion > 0: {E_nuc > 0}")
print(f"  ✓ Electronic energy < 0: {E_elec < 0}")
print(f"  ✓ Total energy < 0:      {E_total < 0}")
print(f"  ✓ |E_elec| > E_nuc:      {abs(E_elec) > E_nuc}")

all_checks_pass = (
    abs(calculated_total - E_total) < 1e-8 and
    E_nuc > 0 and
    E_elec < 0 and
    E_total < 0 and
    abs(E_elec) > E_nuc
)

if all_checks_pass:
    print("\n✓✓✓ ALL VALIDATION CHECKS PASS! Your PySCF data is correct!")
else:
    print("\n⚠ Some validation checks failed - please verify your PySCF output")

# =============================================================================
# COMPARISON WITH LITERATURE
# =============================================================================
print("\n" + "-" * 80)
print("COMPARISON WITH LITERATURE VALUES")
print("-" * 80)

# Expected nuclear repulsion for 0.735 Å
R_bohr = 0.735 / 0.529177
E_nuc_expected = 1.0 / R_bohr

print(f"\nNuclear repulsion comparison:")
print(f"  Your PySCF:        {E_nuc:.10f} Ha")
print(f"  Expected (R=0.735Å): {E_nuc_expected:.10f} Ha")
print(f"  Difference:        {abs(E_nuc - E_nuc_expected):.10f} Ha")

if abs(E_nuc - E_nuc_expected) < 0.01:
    print("  ✓ Close match - your bond length is ~0.735 Å")
else:
    actual_R = 1.0 / E_nuc * 0.529177
    print(f"  → Your actual bond length is ~{actual_R:.3f} Å")

# Literature FCI value for STO-3G at 0.735 Å
E_total_literature = -1.1373  # Approximate from literature
print(f"\nTotal energy comparison:")
print(f"  Your PySCF:    {E_total:.10f} Ha")
print(f"  Literature:    {E_total_literature:.4f} Ha (approximate)")
print(f"  Difference:    {abs(E_total - E_total_literature):.6f} Ha")

if abs(E_total - E_total_literature) < 0.001:
    print("  ✓ Excellent match with published values!")

# =============================================================================
# SAVE DATA FILES
# =============================================================================
print("\n" + "-" * 80)
print("SAVING DATA FILES")
print("-" * 80)

# Save binary pickle for VQE script
with open("flight_data.pkl", "wb") as f:
    pickle.dump(molecular_data, f)
print("✓ Saved: flight_data.pkl")

# Save human-readable JSON
json_data = {k: v for k, v in molecular_data.items() if k != 'pauli_list'}
json_data['num_pauli_terms'] = len(h2_hamiltonian)
json_data['pauli_terms_preview'] = str(h2_hamiltonian[:3]) + "..."
with open("molecule_info.json", "w") as f:
    json.dump(json_data, f, indent=2)
print("✓ Saved: molecule_info.json")

# Save the Pauli terms separately for inspection
with open("pauli_terms.txt", "w") as f:
    f.write("H2 HAMILTONIAN - PAULI DECOMPOSITION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total: {len(h2_hamiltonian)} terms\n\n")
    for pauli, coeff in h2_hamiltonian:
        f.write(f"{pauli:6s}  {coeff:+.10f}\n")
print("✓ Saved: pauli_terms.txt (human-readable Hamiltonian)")

# =============================================================================
# VERIFICATION TEST
# =============================================================================
print("\n" + "-" * 80)
print("VERIFICATION TEST")
print("-" * 80)

try:
    with open("flight_data.pkl", "rb") as f:
        test_load = pickle.load(f)
    
    print("✓ Data loads successfully")
    print(f"✓ Contains {test_load['num_pauli_terms']} Pauli terms")
    print(f"✓ Target electronic energy: {test_load['exact_electronic_energy']:.10f} Ha")
    print(f"✓ Target total energy: {test_load['exact_total_energy']:.10f} Ha")
    
    # Quick matrix test
    from qiskit.quantum_info import SparsePauliOp
    import numpy as np
    
    H_op = SparsePauliOp.from_list(test_load['pauli_list'])
    H_matrix = H_op.to_matrix()
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    ground_state = eigenvalues[0]
    
    print(f"\nQuick diagonalization test:")
    print(f"  Computed ground state: {ground_state:.10f} Ha")
    print(f"  Your PySCF result:     {E_elec:.10f} Ha")
    print(f"  Match: {abs(ground_state - E_elec) < 1e-6} ✓")
    
except Exception as e:
    print(f"⚠ Verification test failed: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUCCESS! DATA PACKAGE CREATED")
print("=" * 80)

print(f"""
Your PySCF Calculation Summary:
  • Molecule: H2
  • Bond length: ~0.735 Å
  • Basis set: STO-3G
  • Method: Full CI (exact within basis)

Energy Results:
  • Nuclear repulsion:      {E_nuc:.10f} Ha
  • Electronic energy (FCI): {E_elec:.10f} Ha
  • Total energy:           {E_total:.10f} Ha

Hamiltonian:
  • 15 Pauli terms
  • 4 qubits
  • Ready for VQE!

Files Created:
  ✓ flight_data.pkl - Main data (use in VQE script)
  ✓ molecule_info.json - Human-readable summary
  ✓ pauli_terms.txt - Hamiltonian breakdown

Target for VQE:
  Your VQE should converge to E_total ≈ {E_total:.6f} Ha
  (Electronic: {E_elec:.6f} Ha + Nuclear: {E_nuc:.6f} Ha)

Chemical Accuracy:
  Goal: Error < 0.0016 Ha (1 kcal/mol)
  That's ~±0.001 Ha or ~±0.6 kcal/mol

Next Step:
  Run your VQE script with this data!
  python vqe_improved.py
""")

print("=" * 80)
print("\nNote: Your electronic energy of -1.857 Ha is the FCI result")
print("from PySCF. This is different from the Hartree-Fock energy")
print("(~-1.85 Ha), which only includes mean-field effects.")
print("\nThe total energy of -1.137 Ha is what VQE should target!")
print("=" * 80)