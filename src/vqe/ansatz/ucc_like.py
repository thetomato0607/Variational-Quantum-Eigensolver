"""Chemistry-inspired UCCSD ansatz."""

from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_uccsd_ansatz(problem):
    """Build a UCCSD ansatz starting from the Hartree-Fock state.

    Args:
        problem: Electronic-structure problem returned by ``get_h2_hamiltonian``.
    """
    mapper = JordanWignerMapper()
    init_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    return UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper, initial_state=init_state)