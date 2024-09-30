from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

setup(
    name='tselect',
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
        'pycatch22>=0.4.2',
        'scikit_learn',
        'sktime>=0.23.1',
        'scipy~=1.10.1',
    ],
    ext_modules=cythonize([
            Extension(
                'tselect.rank_correlation.spearman', ['tselect/rank_correlation/spearman.pyx'],
                include_dirs=[np.get_include()]
            ),]
            ),
)
