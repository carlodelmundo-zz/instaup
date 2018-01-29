#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
#   A tool to transfer a pretrained ResNet network with the exception of FC layers.
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet


def _copy(dst_state_dict, src_state_dict, strict=True):
    """Copies all parameters from src to dst dictionaries. Skips parameters
    starting with 'fc'."""
    for name, param in src_state_dict.items():
        if name.startswith("fc"):
            continue

        if name in dst_state_dict:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                dst_state_dict[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While _copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'
                    .format(name, dst_state_dict[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        missing = set(dst_state_dict.keys()) - set(src_state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def _transfer_parameters(dst_network, src_network):
    _copy(dst_network.state_dict(), src_network.state_dict())


def _difference(a, b):
    '''Returns a scalar with the sum of absolute value between a and b'''
    return np.sum(np.fabs(a-b))


def _validate_parameters(dst_network, src_network):
    dst_state_dict = dst_network.state_dict()
    src_state_dict = src_network.state_dict()
    assert set(dst_state_dict.keys()) == set(src_state_dict.keys())
    for name, _ in src_state_dict.items():
        if name.startswith("fc"):
            continue
        else:
            dst_weights = dst_state_dict[name].numpy().flatten()
            src_weights = src_state_dict[name].numpy().flatten()
            assert (_difference(dst_weights, src_weights) < len(dst_weights) * 1e-6)


if __name__ == "__main__":
    classification_network = resnet.resnet18(pretrained=True)
    regression_network = resnet.resnet18(num_classes=1)
    # These two networks must be identical in structure.
    _transfer_parameters(regression_network, classification_network)
    _validate_parameters(regression_network, classification_network)
    torch.save(regression_network.state_dict(), "/tmp/regression-model.pkl")
