from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules=[ Extension("LogisticRegression",
              sources=["LogisticRegression.pyx", "logistic_regression.cpp"], language='c++', extra_link_args=["-lz", "-mavx512f"], include_dirs=[numpy.get_include()])]

setup(
  name = "LogisticRegression",
  cmdclass = {"build_ext": build_ext},
  ext_modules = cythonize(ext_modules)
)