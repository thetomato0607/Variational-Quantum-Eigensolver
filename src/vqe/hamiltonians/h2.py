from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_h2_hamiltonian(distance: float):
    """
    Generates the H2 Hamiltonian for a specific bond distance.
    Returns: (qubit_operator, problem)
    The problem object contains: nuclear_repulsion_energy, num_particles, num_spatial_orbitals
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