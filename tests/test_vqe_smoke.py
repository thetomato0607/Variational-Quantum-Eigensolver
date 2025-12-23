import unittest
from vqe.vqe_runner import VQERunner
from vqe.hamiltonians.tfim import get_tfim_hamiltonian
from vqe.ansatz.hardware_efficient import get_twolocal_ansatz
from vqe.optimizers.scipy_opt import get_optimizer

class TestVQEPipeline(unittest.TestCase):
    
    def test_simple_tfim_run(self):
        # 1. Setup small problem
        hamiltonian = get_tfim_hamiltonian(2)
        ansatz = get_twolocal_ansatz(2, reps=1)
        optimizer = get_optimizer("COBYLA", maxiter=10)
        
        # 2. Run
        runner = VQERunner(hamiltonian, ansatz, optimizer)
        result = runner.run()
        
        # 3. Verify output structure
        self.assertIn("optimal_value", result)
        self.assertIn("history", result)
        self.assertTrue(len(result["history"]) > 0)

if __name__ == '__main__':
    unittest.main()