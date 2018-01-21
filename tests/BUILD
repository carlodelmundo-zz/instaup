# Author: Carlo C. del Mundo <carlo@xnor.ai>
#
package(default_visibility = ["//visibility:public"])

py_test(
    name = "regression_dataset_test",
    srcs = ["regression_dataset_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    data = [
        "//tests/testdata:images",
        "//tests/testdata:json",
    ],
    deps = [
        "//core:regression_dataset"
    ]
)
