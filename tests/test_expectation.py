"""Tests for the estimator backends."""

import unittest
from vqe.backends.ideal import get_ideal_estimator

class TestBackends(unittest.TestCase):
    """Estimator factory smoke tests."""
    
    def test_ideal_estimator(self):
        """The ideal estimator can be constructed (``run`` is not exercised)."""
        est = get_ideal_estimator()
        self.assertIsNotNone(est)

if __name__ == '__main__':
    unittest.main()