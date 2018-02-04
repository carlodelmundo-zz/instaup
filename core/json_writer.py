# Author: Carlo C. del Mundo <cdel@cs.washington.edu>
# A module to generate a dataset.json file in a directory containing images.
from core import utils
import json
import os

def _path_to_dictionary(path):
    return {"filename": path, "score" : 0.0}

def _json_entries(image_paths):
    """Returns a JSON string listing all image entries in @image_paths."""
    entries = []
    for image_path in image_paths:
        entries.append(_path_to_dictionary(image_path))
    return json.dumps({"entries": entries}, sort_keys=True, indent=4)

def write_json_dataset(dir_path):
    """Creates a dataset.json file in @dir_path enumerating image entries in
    @dir_path."""
    image_paths = utils.get_image_paths(dir_path, absolute=False)
    output_filename = os.path.join(dir_path, "dataset.json")
    with open(output_filename, "w") as f:
        f.write(_json_entries(image_paths))
