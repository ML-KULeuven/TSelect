from setuptools import setup, find_packages

setup(
    name='tsfilter',
    version='1.0.0',
    packages=find_packages(),
    setup_requires=[
        'numpy>=1.22.4'
    ],
    install_requires=[
        'matplotlib>=3.6.2',
        'pandas>=1.5.2',
        'pycatch22>=0.4.2',
        'scikit_learn>=1.1.3',
        'setuptools>=65.5.1',
        'sktime>=0.14.1',
    ],
)