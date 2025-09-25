import os
import shutil

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Distribution
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# Path to your src folder
SRC_DIR = Path(__file__).parent / "src"

def build() -> None:
    ext_modules = cythonize([
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
            ])

    distribution = Distribution({
        "name": "package",
        "ext_modules": ext_modules
    })

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = SRC_DIR / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
