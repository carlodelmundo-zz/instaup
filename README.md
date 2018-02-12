# What is this?

This is the official repository for InstaUp! written in Python 3.
Supported primarily with Ubuntu 16.04 systems. 

You must have the following installed to run tests and executables:

1. [PyTorch](http://pytorch.org/). It's OK if you don't have a GPU installed.
2. [Bazel](https://bazel.build/). This is our build system and helps organize
   models and our test infrastructure.
3. [Flask](http://flask.pocoo.org/). Do `pip3 install Flask --user` to install.


## Testing

Download the [sample
dataset](https://homes.cs.washington.edu/~cdel/contentbooster/sample_dataset.zip) and
place it under `/opt/datasets/`.


Then, to run tests:

```bash
./run_tests.py
```
