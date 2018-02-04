#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
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
