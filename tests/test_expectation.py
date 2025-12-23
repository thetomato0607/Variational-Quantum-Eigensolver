import unittest
from vqe.backends.ideal import get_ideal_estimator

class TestBackends(unittest.TestCase):
    
    def test_ideal_estimator(self):
        est = get_ideal_estimator()
        self.assertIsNotNone(est)

if __name__ == '__main__':
    unittest.main()