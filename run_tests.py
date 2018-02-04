#!/usr/bin/env python3

import argparse
import os
import platform
import sh
import signal
import subprocess
import sys

TESTS = [
    "//tests:inference_test",
    "//tests:regression_dataset_test",
    "//tests:utils_test",
]


def run_bazel(command, args, targets):
    for target in targets:
        subprocess.check_call(["bazel", command] + args + [target])


def run_tests():
    # Quoted and interspersed by "+"
    formatted_packages = " + ".join(["\"{}\"".format(x) for x in TESTS])
    QUERY = "kind(test, {})".format(formatted_packages)
    targets = sh.bazel.query(QUERY)
    targets = targets.split()

    extra_flags = []
    if platform.system() == "Darwin":
        extra_flags += ["--config", "macos"]

    run_bazel("run", ["-c", "dbg"] + extra_flags, targets)


def main(args=None):
    try:
        run_tests()
    except (subprocess.CalledProcessError, sh.ErrorReturnCode,
            KeyboardInterrupt) as err:
        print("Error. Tests aborted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
