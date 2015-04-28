#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy.distutils.misc_util

KMeans_ext = Extension("KMeans._KMeans", ["KMeans/_KMeans.cpp", "KMeans/KMeans.cpp"])

setup(
    ext_modules=[KMeans_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
