from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
from pathlib import Path

# Path to your src folder
SRC_DIR = Path(__file__).parent / "src"

setup(
    name='tselect',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # tells setuptools to look under src/
    python_requires='>=3.10',
    setup_requires=[
        'setuptools>=80.9.0',
        'Cython>=3.1.4',
        'numpy>=2.3.3'
    ],
    install_requires=[
        'six>=1.17.0',
        'graphviz>=0.21',
        'matplotlib>=3.10.6',
        'pandas>=2.3.2',
        'pycatch22>=0.4.5',
        'scikit_learn>=1.7.2',
        'sktime>=0.38.5',
        'scipy>=1.16.2',
        'statsmodels>=0.14.5',
        'Pint>=0.25',
    ],
    extras_require={'test': [
        'pytest',
    ]},
    ext_modules=cythonize([
            Extension(
                'tselect.rank_correlation.spearman',
                [str(SRC_DIR / 'tselect/rank_correlation/spearman.pyx')],
                include_dirs=[np.get_include()],
            ),
            Extension(
                'tsfuse.data.df', [str(SRC_DIR / 'tsfuse/data/df.pyx')],
                include_dirs=[np.get_include()]
        ),
            Extension(
                'tsfuse.transformers.calculators.cstatistics', [str(SRC_DIR / 'tsfuse/transformers/calculators/cstatistics.pyx')],
                include_dirs=[np.get_include()]
        ),
            Extension(
                'tsfuse.transformers.calculators.queries', [str(SRC_DIR / 'tsfuse/transformers/calculators/queries.pyx')],
                include_dirs=[np.get_include()]
        ),
            ]),
)
