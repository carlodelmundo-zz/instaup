# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
#   A regression dataset returns a list of (image_path, score) pairs populated
#   from a JSON file.
from torch.utils.data import dataset
import os
import json


def _read_json(json_path):
    """Returns a dictionary representing the JSON data in @json_path"""
    json_path = os.path.expanduser(json_path)
    with open(json_path) as json_data:
        return json.load(json_data)

# TODO (carlo): do basic error checking for (image_path, score) pairs.
def _images_and_scores(json_path):
    json_data = _read_json(json_path)
    images_and_scores = []
    for entry in json_data["entries"]:
        images_and_scores.append((entry["filename"], entry["score"]))
    return images_and_scores

class RegressionDataset(dataset.Dataset):
    def __init__(self, json_path):
        self.images_and_scores = _images_and_scores(json_path)

    def __len__(self):
        return len(self.images_and_scores)

    def __getitem__(self, index):
        return self.images_and_scores[index]
