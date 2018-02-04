# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
from core import inference
from core import utils
import unittest
import numpy as np
import sys

_SAMPLE_DATASET_PATH = "/opt/datasets/sample_dataset/"


class TestInference(unittest.TestCase):
    def test_infer_monotonic(self):
        '''Checks if the returned scores are monotonically decreasing.'''
        computed_output = inference.infer(_SAMPLE_DATASET_PATH, num_results=4)
        current_score = sys.float_info.max
        for index, score in computed_output:
            self.assertGreaterEqual(current_score, score)
            current_score = score


if __name__ == '__main__':
    unittest.main()
