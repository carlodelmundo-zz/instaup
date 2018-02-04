# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
from core import utils
import unittest
import numpy as np


class TestUtils(unittest.TestCase):
    def test_top_k_simple(self):
        reference_scores = np.array([0.2, 0.8, 0.1, 0.0, -0.08])
        reference_output = np.array([1, 0, 2, 3, 4])
        for k in range(5):
            computed_output = utils.top_k(reference_scores, k)
            self.assertTrue(
                utils.check_equal(computed_output, reference_output[:k]))


if __name__ == '__main__':
    unittest.main()
