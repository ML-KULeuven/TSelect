from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

setup(
    name='tselect',
    version='0.1.2',
    packages=find_packages(),
    python_requires='>=3.10, <3.11',
    setup_requires=[
        'setuptools>=70.0.0',
        'Cython>=3.0.11',
        'numpy>=1.26.4'
    ],
    install_requires=[
        'matplotlib>=3.10.0',
        'pandas>=1.5.3',
        'pycatch22>=0.4.2',
        'scikit_learn~=1.3.2',
        'sktime>=0.23.1',
        'scipy~=1.15.1',
    ],
    ext_modules=cythonize([
            Extension(
                'tselect.rank_correlation.spearman', ['tselect/rank_correlation/spearman.pyx'],
                include_dirs=[np.get_include()]
            ),]
            ),
)
