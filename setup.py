#!/usr/bin/env python

import os
from distutils.spawn import spawn
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy.distutils.misc_util
import sys

__version__ = '0.1'

extra_compile_args = ["-lpython"]
extra_compile_args = os.environ.get('NVCCFLAGS', '').split() + extra_compile_args

libraries = ['cublas', 'python']

KMeans_ext = Extension("PyCUDAML.KMeans._KMeans",
                       sources=["PyCUDAML/KMeans/_KMeans.cu",
                                "PyCUDAML/KMeans/KMeans.cu"],
                       libraries=libraries,
                       extra_compile_args=extra_compile_args)

# Borrowed from cudamat setup.py
# https://github.com/cudamat/cudamat/blob/master/setup.py
class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0, display=None):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        spawn(cmd, search_path, verbose, dry_run)

setup(
    name='PyCUDAML',
    author="Yan Zhao",
    version=__version__,
    url='https://github.com/zhaoyan1117/PyCUDAML',
    include_package_data=True,
    packages=find_packages(),
    ext_modules=[KMeans_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    cmdclass={'build_ext': CUDA_build_ext}
)
