"""Tests for the Hamiltonian builders."""

import unittest
from vqe.hamiltonians.h2 import get_h2_hamiltonian
from vqe.hamiltonians.tfim import get_tfim_hamiltonian

class TestHamiltonians(unittest.TestCase):
    """Hamiltonian construction."""
    
    def test_h2_generation(self):
        """H2 in STO-3G at equilibrium maps to 4 qubits."""
        # Test H2 at equilibrium
        qubit_op, _ = get_h2_hamiltonian(0.735)
        self.assertIsNotNone(qubit_op)
        # H2 STO-3G should be 4 qubits
        self.assertEqual(qubit_op.num_qubits, 4)

    def test_tfim_generation(self):
        """A 2-site TFIM has the right width and contains a ZZ coupling term."""
        # Test 2-qubit Ising model
        op = get_tfim_hamiltonian(num_qubits=2)
        self.assertEqual(op.num_qubits, 2)
        # Should have terms like ZZ and X
        op_list = op.to_list()
        self.assertTrue(any("ZZ" in x[0] for x in op_list))

if __name__ == '__main__':
    unittest.main()