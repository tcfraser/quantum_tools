from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        language='c',
        name='quantum_tools.utilities.number_system_tools',
        sources=['quantum_tools/utilities/number_system_tools.pyx']
        )
]

setup(
    ext_modules = cythonize(extensions),
)