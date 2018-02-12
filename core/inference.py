# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
from core import json_writer
from core import regression_dataset
from core import utils
import numpy as np
import operator
import torch
from torch.autograd import Variable
from torchvision.models import resnet
from torchvision import transforms

_MODEL_PATH = "external/selfies_resnet18/file/regression-model.pkl"


def _loader(dataset_path, batch_size):
    '''Returns a DataLoader that emits (image_data, score) tuples. Data is not
    shuffled and crops are always center cropped.'''
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.CenterCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    dataset = regression_dataset.RegressionDataset(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return data_loader


def infer(dataset_path, num_results):
    '''Returns a list of (index, score) tuples corresponding to the top-k
    highest scored images by the model.'''
    data_loader = _loader(dataset_path, batch_size=4)
    if num_results > len(data_loader.dataset):
        raise ValueError(
            "num_results ({}) may not be greater than the cardinality of the dataset ({})".
            format(num_results, len(data_loader.dataset)))
    net = resnet.resnet18(num_classes=1)
    net.load_state_dict(torch.load(_MODEL_PATH))
    net.eval()
    scores = np.zeros(0)
    for image_data, _ in data_loader:
        inputs = Variable(image_data)
        outputs = net(inputs).data.numpy().flatten()
        scores = np.append(scores, outputs)
    top_indices = utils.top_k(scores, num_results)
    return list(zip(top_indices, scores[top_indices]))


def score_image_directory(image_dir, num_results):
    """Returns the top-@num_results image entries from @image_dir as
    (image_path,score) pairs."""
    # Creates and places a dataset.json file inside @image_dir, so we can feed
    # these images to the inference pipeline.
    json_writer.write_json_dataset(image_dir)
    idx_and_scores = infer(image_dir, num_results)
    image_paths = utils.get_image_paths(image_dir, absolute=True)
    results = []
    for idx, score in idx_and_scores:
        results.append((image_paths[idx], score))
    # Sort entries in descending order by score.
    results.sort(key=operator.itemgetter(1), reverse=True)
    return results
