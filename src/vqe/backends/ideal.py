from qiskit.primitives import Estimator

def get_ideal_estimator():
    # Note: qiskit_algorithms.VQE (pinned to 0.3.1) calls estimator.run() with
    # the legacy V1 positional signature (circuits, observables, parameters),
    # which the V2 StatevectorEstimator does not accept (it expects PUBs).
    # The V1 Estimator is deprecated but still bundled with qiskit==1.2.4 and
    # is what qiskit-algorithms 0.3.1 expects.
    return Estimator()