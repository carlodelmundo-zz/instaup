# Author: Carlo C. del Mundo <carlo@xnor.ai>
#
package(default_visibility = ["//visibility:public"])

py_library(
    name = "regression_dataset",
    srcs = ["regression_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "regression_dataset_test",
    srcs = ["regression_dataset_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    data = [
        "//core/testdata:images",
        "//core/testdata:json",
    ],
    deps = [
        ":regression_dataset"
    ]
)
