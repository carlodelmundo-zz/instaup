# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
#   A regression dataset returns a list of (image_path, score) pairs populated
#   from a JSON file.
from torch.utils.data import dataset
from core import utils
from PIL import Image
import os
import json


def _read_json(json_path):
    """Returns a dictionary representing the JSON data in @json_path"""
    json_path = os.path.expanduser(json_path)
    with open(json_path) as json_data:
        return json.load(json_data)


def _validate_filename(filename):
    if not os.path.exists(filename):
        raise ValueError("The filename {} does not exist.".format(filename))
    if not utils.is_image_file(filename):
        raise ValueError(
            "The filename {} is not a valid image file.".format(filename))


def _validate_score(score):
    if score < 0.0:
        raise ValueError("The score {} may not be negative.".format(score))
    if score > 1.0:
        raise ValueError(
            "The score {} may not be greater than 1.0.".format(score))


def _images_and_scores(dataset_path):
    json_data = _read_json(dataset_path + "dataset.json")
    images_and_scores = []
    for entry in json_data["entries"]:
        filename = dataset_path + entry["filename"]
        score = entry["score"]
        _validate_filename(filename)
        _validate_score(score)
        images_and_scores.append((filename, score))
    return images_and_scores


def _path_to_image_data(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class RegressionDataset(dataset.Dataset):
    def __init__(self, dataset_path, transform=None):
        """A dataset is a set of images accompanied with a dataset.json file.
        Transform is a set of torchvision transformations (see:
        https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py)"""
        self.images_and_scores = _images_and_scores(dataset_path)
        self.transform = transform

    def __len__(self):
        return len(self.images_and_scores)

    def __getitem__(self, index):
        img_path, target = self.images_and_scores[index]
        img_data = _path_to_image_data(img_path)
        if self.transform is not None:
            img_data = self.transform(img_data)
        return img_data, target
