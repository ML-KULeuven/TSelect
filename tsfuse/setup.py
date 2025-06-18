from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='tsfuse',
    version='0.2.2',
    packages=find_packages(),
    python_requires='>=3.10, <3.11',
    include_package_data=True,
    package_data={},
    setup_requires=[
        'setuptools',
        'Cython>=3.0.5',
        'numpy>=1.22.4'
    ],
    install_requires=[
        'six>=1.17.0',
        'graphviz>=0.20.2',
        'scipy>=1.15.1',
        'scikit-learn>=1.3.2',
        'statsmodels>=0.14.0',
        'Pint>=0.21.1',
        'matplotlib==3.10.0',
        'pandas>=1.5.3',
        'sktime>=0.23.1',
    ],
    extras_require={'test': [
        'pytest',
        'pandas>=0.24.2'
    ]},
    ext_modules=cythonize([
        Extension(
            'tsfuse.data.df', ['tsfuse/data/df.pyx'],
            include_dirs=[np.get_include()]
        ),
        Extension(
            'tsfuse.transformers.calculators.*', ['tsfuse/transformers/calculators/*.pyx'],
            include_dirs=[np.get_include()]
        ),
    ]),
)
