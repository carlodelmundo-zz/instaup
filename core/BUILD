# Author: Carlo C. del Mundo <carlo@xnor.ai>
#
package(default_visibility = ["//visibility:public"])

py_library(
    name = "regression_dataset",
    srcs = ["regression_dataset.py"],
    deps = [
        ":utils",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "inference",
    srcs = ["inference.py"],
    srcs_version = "PY3",
    data = [
        "@resnet18//file",
    ],
    deps = [
        ":regression_dataset",
        ":utils",
    ],
)

py_library(
    name = "json_writer",
    srcs = ["json_writer.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)
