# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
from core import regression_dataset
import unittest
import torchvision
from torchvision import transforms

_DATASET_PATH = "/opt/datasets/sample_dataset/"


def _is_normalized(data):
    for value in data:
        if value < 0.0:
            return False
        elif value > 1.0:
            return False
    return True


class TestRegressionDataset(unittest.TestCase):
    def test_length(self):
        dataset = regression_dataset.RegressionDataset(_DATASET_PATH)
        self.assertEqual(len(dataset), 4)

    def test_order_of_entries(self):
        """dataset[k] returns the kth (img_data, score) tuple. Here, we just
        test equality on scores. See the dataset.json file in _DATASET_PATH for
        scores."""
        dataset = regression_dataset.RegressionDataset(_DATASET_PATH)
        self.assertEqual(dataset[0][1], 0.998)
        self.assertEqual(dataset[1][1], 0.734)
        self.assertEqual(dataset[2][1], 0.343)
        self.assertEqual(dataset[3][1], 0.123)

    def test_data_normalized(self):
        """Checks if the image data provided by this dataset is in [0.0,1.0]
        given the ToTensor() transformation."""
        dataset = regression_dataset.RegressionDataset(_DATASET_PATH,
                                                       transforms.ToTensor())
        for img_data, _ in dataset:
            self.assertTrue(_is_normalized(img_data.numpy().flatten()))


if __name__ == '__main__':
    unittest.main()
