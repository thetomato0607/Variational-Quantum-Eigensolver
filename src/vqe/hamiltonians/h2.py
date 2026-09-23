"""Molecular-hydrogen Hamiltonian (PySCF, STO-3G, Jordan-Wigner)."""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_h2_hamiltonian(distance: float):
    """Build the H2 qubit Hamiltonian at a given bond length.

    Args:
        distance: H-H separation in angstrom.

    Returns:
        ``(qubit_op, problem)``. ``qubit_op`` is the electronic Hamiltonian
        only; add ``problem.nuclear_repulsion_energy`` to get total energies.
    """
    driver = PySCFDriver(
        atom=f"H 0.0 0.0 0.0; H 0.0 0.0 {distance}",
        charge=0,
        spin=0,
        basis='sto-3g'
    )
    problem = driver.run()
    hamiltonian = problem.hamiltonian.second_q_op()
    qubit_op = JordanWignerMapper().map(hamiltonian)

    return qubit_op, problem