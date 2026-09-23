"""Tests for the ansatz factories."""

import unittest
from vqe.ansatz.hardware_efficient import get_twolocal_ansatz

class TestAnsatz(unittest.TestCase):
    """TwoLocal ansatz construction."""
    
    def test_twolocal(self):
        """TwoLocal has the requested width and trainable parameters."""
        ansatz = get_twolocal_ansatz(num_qubits=4, reps=2)
        self.assertEqual(ansatz.num_qubits, 4)
        # Check if it has parameters
        self.assertTrue(ansatz.num_parameters > 0)

if __name__ == '__main__':
    unittest.main()