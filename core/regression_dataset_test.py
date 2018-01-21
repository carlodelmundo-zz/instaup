# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
import regression_dataset
import unittest

_SAMPLE_TRAINING_SET = "./core/testdata/json/sample_training_set.json"

class TestRegressionDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = regression_dataset.RegressionDataset(_SAMPLE_TRAINING_SET)

    def test_length(self):
        self.assertEqual(len(self.dataset), 4)

    def test_order_of_entries(self):
        self.assertEqual(self.dataset[0], ("ILSVRC2012_val_00000523.JPEG", 0.998))
        self.assertEqual(self.dataset[1], ("ILSVRC2012_val_00000539.JPEG", 0.734))
        self.assertEqual(self.dataset[2], ("ILSVRC2012_val_00000507.JPEG", 0.343))
        self.assertEqual(self.dataset[3], ("ILSVRC2012_val_00000524.JPEG", 0.123))
        

if __name__ == '__main__':
    unittest.main()
