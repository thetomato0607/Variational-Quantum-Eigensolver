from typing import Any, Dict, List

from qiskit_algorithms import VQE

from .backends.ideal import get_ideal_estimator

class VQERunner:
    def __init__(
        self,
        hamiltonian,
        ansatz,
        optimizer,
        estimator=None,
        *,
        verbose: bool = True,
        print_every: int = 10,
    ):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.estimator = estimator if estimator else get_ideal_estimator()
        self.history: List[float] = []
        self.verbose = verbose
        self.print_every = max(1, print_every)

    def callback(self, eval_count, params, mean, std):
        """Standard callback to track convergence."""
        # Store just the energy value for plotting
        self.history.append(mean)
        if self.verbose and eval_count is not None:
            if eval_count % self.print_every == 0 or eval_count == 1:
                print(f"   Iter {eval_count}: Energy = {mean:.5f} Ha")

    def run(self) -> Dict[str, Any]:
        """Executes the VQE algorithm."""
        self.history = []  # Reset history
        
        vqe = VQE(
            estimator=self.estimator,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            callback=self.callback
        )
        
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        
        return {
            "optimal_value": result.eigenvalue.real,
            "optimal_params": result.optimal_point,
            "history": self.history
        }
