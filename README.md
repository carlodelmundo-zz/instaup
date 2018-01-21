# What is this?

This is the official repository for Content Booster.

## Requirements

You must have the following installed to run tests and executables:

1. [PyTorch](http://pytorch.org/). It's OK if you don't have a GPU installed.
2. [Bazel](https://bazel.build/). This is our build system and helps organize models and our test infrastructure.

## Testing

To run the test suite, simply do:

*./run_tests.py*

## JSON Format

An example JSON file for training models is excerpted below:

```json
{
    "description": "Sample image dataset for (image,score) pairs.",
    "path": "/opt/datasets/sample_images/",
    "entries": [
        {
            "filename": "ILSVRC2012_val_00000523.JPEG",
            "score": 0.998
        },
        {
            "filename": "ILSVRC2012_val_00000539.JPEG",
            "score": 0.734
        },
        {
            "filename": "ILSVRC2012_val_00000507.JPEG",
            "score": 0.343
        },
        {
            "filename": "ILSVRC2012_val_00000524.JPEG",
            "score": 0.123
        }
    ]
}
```

Here, image files are locally stored in /opt/datasets/sample\_images.
