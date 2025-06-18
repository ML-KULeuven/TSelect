from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy


# `python setup_queries.py build_ext --inplace`

setup(
    name="queries_lib",
    ext_modules=cythonize(
        Extension(
            "queries",
            sources=["queries.pyx"],
            extra_compile_args=["-O3", "-mavx", "-mavx2"],
            include_dirs=[numpy.get_include()],
        ),
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)

# setup(
#     name="queries_lib",
#     ext_modules=[
#         Extension("queries",
#             sources=["queries.pyx"],
#             extra_compile_args=["-O3", "-mavx", "-mavx2"],
#             include_dirs=[numpy.get_include()]
#         )
#     ],
#     zip_safe=False,
#     cmdclass = {'build_ext': build_ext}
# )
