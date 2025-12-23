from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_uccsd_ansatz(problem):
    mapper = JordanWignerMapper()
    init_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    return UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper, initial_state=init_state)