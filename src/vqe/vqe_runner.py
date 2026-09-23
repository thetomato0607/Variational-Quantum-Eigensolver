"""Thin wrapper around qiskit-algorithms' VQE that records the energy history."""

from typing import Any, Dict, List

from qiskit_algorithms import VQE

from .backends.ideal import get_ideal_estimator

class VQERunner:
    """Run VQE and record the energy at every optimizer evaluation.

    Args:
        hamiltonian: Qubit operator to minimise (electronic part only for H2).
        ansatz: Parameterised circuit.
        optimizer: A qiskit-algorithms optimizer.
        estimator: V1 Estimator; defaults to ``get_ideal_estimator()``.
        verbose: Print progress while optimising.
        print_every: Print interval in evaluations (clamped to at least 1).
    """
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
        """Record one energy evaluation and optionally print progress.

        qiskit-algorithms passes estimator metadata as the fourth argument; it is
        named ``std`` here but unused.
        """
        self.history.append(mean)
        if self.verbose and eval_count is not None:
            if eval_count % self.print_every == 0 or eval_count == 1:
                print(f"   Iter {eval_count}: Energy = {mean:.5f} Ha")

    def run(self) -> Dict[str, Any]:
        """Run VQE from scratch, resetting ``history``.

        Returns:
            Dict with ``optimal_value`` (electronic energy; nuclear repulsion not
            included), ``optimal_params`` and ``history``.
        """
        self.history = []
        
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
