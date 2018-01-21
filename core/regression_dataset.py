# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
#   A regression dataset returns a list of (image_path, score) pairs populated
#   from a JSON file.
from torch.utils.data import dataset
from core import utils
import os
import json


def _read_json(json_path):
    """Returns a dictionary representing the JSON data in @json_path"""
    json_path = os.path.expanduser(json_path)
    with open(json_path) as json_data:
        return json.load(json_data)

def validate_filename(filename):
    if not os.path.exists(filename):
        raise ValueError("The filename {} does not exist.".format(filename))
    if not utils.is_image_file(filename):
        raise ValueError("The filename {} is not a valid image file.".format(filename))

def validate_score(score):
    if score < 0.0:
        raise ValueError("The score {} may not be negative.".format(score))
    if score > 1.0:
        raise ValueError("The score {} may not be greater than 1.0.".format(score))


# TODO (carlo): do basic error checking for (image_path, score) pairs.
def _images_and_scores(dataset_path):
    json_data = _read_json(dataset_path + "dataset.json")
    images_and_scores = []
    for entry in json_data["entries"]:
        filename = dataset_path + entry["filename"]
        score = entry["score"]
        validate_filename(filename)
        validate_score(score)
        images_and_scores.append((filename, score))
    return images_and_scores

class RegressionDataset(dataset.Dataset):
    def __init__(self, json_path):
        self.images_and_scores = _images_and_scores(json_path)

    def __len__(self):
        return len(self.images_and_scores)

    def __getitem__(self, index):
        return self.images_and_scores[index]
