#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]


def is_image_file(filename):
    """Return True if the filename ends with a known image extension"""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in _IMG_EXTENSIONS)
