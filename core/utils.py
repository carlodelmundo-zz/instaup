#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
import os

_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]

def is_image_file(filename):
    """Return True if the filename ends with a known image extension"""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in _IMG_EXTENSIONS)


def top_k(scores, k):
    '''Returns the indices of the top-k scores.'''
    return (-scores).argsort()[:k]


def check_equal(a, b):
    '''True if the two lists are equal.'''
    return len(a) == len(b) and sorted(a) == sorted(b)

def get_image_paths(dir_path, absolute=True):
    """Returns a list of filepaths of images in @dir_path. The filepath
    returned is an absolute filepath by default."""
    dir_path = os.path.expanduser(dir_path)
    if not os.path.isdir(dir_path):
        raise ValueError(
            "Expected @dir_path = {} to be a directory.".format(dir_path))
    image_paths = []
    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                if absolute:
                    image_paths.append(os.path.join(root, fname))
                else:
                    image_paths.append(fname)
    return image_paths
