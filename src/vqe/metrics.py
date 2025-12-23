def compute_chemical_accuracy(exact_energy, computed_energy):
    """Returns True if error is within 1 kcal/mol (0.0016 Ha)."""
    error = abs(exact_energy - computed_energy)
    return error < 0.0016

def hartree_to_kcal(energy_ha):
    return energy_ha * 627.5