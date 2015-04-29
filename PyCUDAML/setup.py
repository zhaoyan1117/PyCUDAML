#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup, Extension
import numpy.distutils.misc_util

__version__ = '0.1'

KMeans_ext = Extension("KMeans._KMeans",
                       sources=["KMeans/_KMeans.cpp", "KMeans/KMeans.cpp"])

setup(
    name='PyCUDAML',
    version=__version__,
    url='https://github.com/zhaoyan1117/PyCUDAML',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[KMeans_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
