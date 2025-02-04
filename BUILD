# Description:
#   A metrics library for JAX.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(name = "license")

licenses(["notice"])

exports_files(["LICENSE"])

pytype_strict_library(
    name = "metrax",
    srcs = ["__init__.py"],
    deps = [":metrics"],
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        "//third_party/py/clu:metrics",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

py_strict_test(
    name = "metrics_test",
    srcs = [
        "metrics_test.py",
    ],
    deps = [
        ":metrax",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/sklearn",
    ],
)