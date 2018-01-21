#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
# Returns the top-k images and their scores given a path to a dataset.
# Usage:
#   python3 image_selector.py --image_dir /opt/models/images/

import argparse
from core import utils
import os
import numpy as np
import random
import operator


def _get_image_paths(dir_path):
    """Returns a list of image_paths in @dir_path."""
    dir_path = os.path.expanduser(dir_path)
    if not os.path.isdir(dir_path):
        raise ValueError(
            "Expected @dir_path = {} to be a directory.".format(dir_path))
    image_paths = []
    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in sorted(fnames):
            if utils.is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)
    return image_paths


# TODO (carlo): Replace this with actual backend.
def _select_and_score(image_paths, num_results):
    """Returns the top-@num_results entries as (image,score) pairs."""
    # Randomly choose with equal probability.
    top_choices = random.sample(image_paths, num_results)
    # Make up the score.
    scores = np.random.uniform(0.0, 1.0, num_results).tolist()
    results = list(zip(top_choices, scores))
    # Sort entries in descending order by score.
    results.sort(key=operator.itemgetter(1), reverse=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selects the top-k images from a dataset.")
    parser.add_argument(
        "--num_results",
        required=False,
        default=3,
        metavar="N",
        type=int,
        nargs="?",
        help="The number of results to return.")
    parser.add_argument(
        "--image_dir",
        required=True,
        help="The directory where the images exist")
    args = parser.parse_args()

    if args.num_results < 0:
        raise ValueError("args.num_results may not be negative")
    image_paths = _get_image_paths(args.image_dir)
    if args.num_results > len(image_paths):
        raise ValueError(
            "The number of requested results ({}) may not exceed the length of the image set ({})".
            format(args.num_results, len(image_paths)))
    results = _select_and_score(image_paths, args.num_results)
    for image_name, score in results:
        print("{}. Score: {}".format(image_name, score))
