# Author: Carlo C. del Mundo <carlo@xnor.ai>
#
package(default_visibility = ["//visibility:public"])

py_test(
    name = "regression_dataset_test",
    srcs = ["regression_dataset_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//core:regression_dataset"
    ]
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//core:utils"
    ]
)

py_test(
    name = "inference_test",
    srcs = ["inference_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//core:inference",
        "//core:utils"
    ]
)
