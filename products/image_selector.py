#!/usr/bin/env python3
# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
# A command like tool to return the top-k images and their scores given a
# folder containing images.
# Usage:
#   On Ubuntu 16.04:
#       bazel run :image_selector -- --image_dir /opt/models/images/
#   On Mac OS X:
#       bazel run --config macos :image_selector -- --image_dir /opt/models/images/
#
#   Sample Output:
#   INFO: Running command line: bazel-bin/products/image_selector --image_dir /opt/models/images/
#   /opt/models/images/ILSVRC2012_val_00000470.JPEG. Score: 0.33189800767936795
#   /opt/models/images/ILSVRC2012_val_00000517.JPEG. Score: 0.08722822272474651
#   /opt/models/images/ILSVRC2012_val_00000228.JPEG. Score: 0.008819753286334775
import argparse
from core import inference
from core import utils


def _validate_args(args):
    if args.num_results < 0:
        raise ValueError("args.num_results may not be negative")
    image_paths = utils.get_image_paths(args.image_dir)
    if args.num_results > len(image_paths):
        raise ValueError(
            "The number of requested results ({}) may not exceed the length of the image set ({})".
            format(args.num_results, len(image_paths)))


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
    _validate_args(args)
    results = inference.score_image_directory(args.image_dir, args.num_results)
    for image_name, score in results:
        print("{}. Score: {}".format(image_name, score))
