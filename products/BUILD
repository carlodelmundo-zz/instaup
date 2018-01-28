# Author: Carlo C. del Mundo <carlo@xnor.ai>
#
package(default_visibility = ["//visibility:public"])

py_binary(
    name = "image_selector",
    srcs = ["image_selector.py"],
    main = "image_selector.py",
    srcs_version = "PY3",
    deps = [
        "//core:utils",
    ],
)

py_binary(
    name = "model_trainer",
    srcs = ["model_trainer.py"],
    main = "model_trainer.py",
    srcs_version = "PY3",
    deps = [
        "//core:regression_dataset",
    ],
)
