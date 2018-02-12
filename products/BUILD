# Author: Carlo C. del Mundo <carlo@xnor.ai>
#
package(default_visibility = ["//visibility:public"])

py_binary(
    name = "image_selector",
    srcs = ["image_selector.py"],
    main = "image_selector.py",
    srcs_version = "PY3",
    deps = [
        "//core:inference",
        "//core:utils",
    ],
)

py_binary(
    name = "model_trainer",
    srcs = ["model_trainer.py"],
    main = "model_trainer.py",
    srcs_version = "PY3",
    data = [
        "@selfies_resnet18//file",
    ],
    deps = [
        "//core:regression_dataset",
    ],
)

py_binary(
    name = "web_demo",
    srcs = ["web_demo.py"],
    main = "web_demo.py",
    srcs_version = "PY3",
    data = [
        ":static",
        ":templates",
    ],
    deps = [
        "//core:inference",
        "//core:utils",
    ],
)

filegroup(
    name = "templates",
    srcs = glob(["templates/**"]),
)

filegroup(
    name = "static",
    srcs = glob(["static/**"]),
)
