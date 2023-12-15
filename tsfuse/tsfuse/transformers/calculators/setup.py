from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


# `python setup.py build_ext --inplace`

setup(
    name="cstatistics_lib",
    ext_modules=[
        Extension("cstatistics",
            sources=["cstatistics.pyx"],
            extra_compile_args=["-O3", "-mavx", "-mavx2"],
            include_dirs=[numpy.get_include()]
        )
    ],
    zip_safe=False,
    cmdclass = {'build_ext': build_ext}
)
