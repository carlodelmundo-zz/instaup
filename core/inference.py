# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
from core import regression_dataset
from core import utils
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.models import resnet
from torchvision import transforms

_MODEL_PATH = "external/resnet18/file/regression-model.pkl"

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
