from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

setup(
    name='tsfilter',
    version='1.0.0',
    packages=find_packages(),
    setup_requires=[
        'setuptools',
        'Cython>=3.0.5',
        'numpy>=1.22.4'
    ],
    install_requires=[
        'matplotlib>=3.7.3',
        'pandas>=1.5.3',
        'pycatch22>=0.4.4',
        'scikit_learn>=1.1.3',
        'sktime>=0.23.1',
        'Deprecated~=1.2.14',
        'scipy~=1.11.3',
    ],
    ext_modules=cythonize([
            Extension(
                'tsfilter.rank_correlation.spearman', ['tsfilter/rank_correlation/spearman.pyx'],
                include_dirs=[np.get_include()]
            ),]
            ),
)
