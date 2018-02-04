#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
from core import regression_dataset
import torch
from torch import nn
from torch import optim
from torchvision.models import resnet
from torchvision import transforms
from torch.autograd import Variable

_SAMPLE_DATASET_PATH = "/opt/datasets/sample_dataset/"
_MODEL_PATH = "external/resnet18/file/regression-model.pkl"


def _loader(dataset_path, batch_size):
    '''Returns a DataLoader that emits (image_data, score) tuples. Dataset is
    randomly shuffled and randomly cropped.'''
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        # TODO (carlo): Figure out a sensible way to do data augmentations here.
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    dataset = regression_dataset.RegressionDataset(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader


def train_network(batch_size, epochs):
    train_loader = _loader(_SAMPLE_DATASET_PATH, batch_size)
    # Typically, num_classes specifies the number of object classes in image
    # classification. Here, we fix num_classes to 1 since we are regressing,
    # and not classifying. The output of an inference is a single number.
    net = resnet.resnet18(num_classes=1)
    net.load_state_dict(torch.load(_MODEL_PATH))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        for image_data, scores in train_loader:
            optimizer.zero_grad()
            inputs, labels = Variable(image_data), Variable(scores.float())
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("Mean squared error (loss) = {}".format(loss.data[0]))


if __name__ == "__main__":
    train_network(batch_size=4, epochs=512)
