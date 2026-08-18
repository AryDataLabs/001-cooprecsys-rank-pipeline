"""
Build Cython extensions.
Usage: python setup.py build_ext --inplace
       OR: python -m pip install -e .
"""
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

ext = Extension(
    name="cooprecsys._cy._cy_predict",
    sources=["cooprecsys/_cy/_cy_predict.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="cooprecsys",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
