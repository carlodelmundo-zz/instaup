#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
# Trains a regression model given a regression dataset. Trained model snapshots
# are saved in _MODEL_SNAPSHOT_DIR.
# Usage:
#  bazel run :model_trainer -- --dataset /opt/datasets/sample_dataset
# Sample output:
#  Mean squared error (loss) = 0.7843878865242004
import argparse
from core import regression_dataset
import os
import sys
import torch
from torch import nn
from torch import optim
from torchvision.models import resnet
from torchvision import transforms
from torch.autograd import Variable

# Which base model to use. This model is a pretrained ResNet-18 model trained
# for classification, but repurposed for regression.
_DEFAULT_MODEL_PATH = "external/selfies_resnet18/file/regression-model-cpu-20180224.pkl"
# Save models in the user's home directory. Only the latest model is saved.
_MODEL_SNAPSHOT_DIR = "~/"


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


def _save_model(model):
    output_dir = os.path.expanduser(_MODEL_SNAPSHOT_DIR)
    output_path = os.path.join(output_dir, "regression-model.pkl")
    print("Snapshotting model to {}".format(output_path))
    # Transfer model back to the CPU so inference engines without GPU support
    # can still run the model.
    model.cpu()
    torch.save(model.state_dict(), output_path)


def train_network(model_path, dataset, batch_size):
    train_loader = _loader(dataset, batch_size)
    # Typically, num_classes specifies the number of object classes in image
    # classification. Here, we fix num_classes to 1 since we are regressing,
    # and not classifying. The output of an inference is a single number.
    net = resnet.resnet18(num_classes=1)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Train indefinitely.
    for epoch in range(sys.maxsize):
        for image_data, scores in train_loader:
            optimizer.zero_grad()
            inputs, labels = Variable(image_data.cuda()), Variable(
                scores.float().cuda())
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("Mean squared error (loss) = {}".format(loss.data[0]))
        if (epoch + 1) % 100 == 0:
            _save_model(net)
            net.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="The directory containing a training dataset. A dataset consists"
        " of images and a dataset.json file")
    parser.add_argument(
        "--model_path",
        required=False,
        default=_DEFAULT_MODEL_PATH,
        help="The path to the model weights (.pkl) file.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise "Training is only supported on devices with CUDA GPUs."
    train_network(
        os.path.expanduser(args.model_path),
        os.path.expanduser(args.dataset),
        batch_size=64)
