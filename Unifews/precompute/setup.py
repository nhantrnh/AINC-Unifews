from distutils.core import setup, Extension
from Cython.Build import cythonize
import eigency

ext = Extension(
    name='prop',
    sources=['prop.pyx'],
    language='c++',
    extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
    include_dirs=[
        ".",
        "/usr/include/eigen3",
    ] + eigency.get_includes(),
)

setup(
    author='nyLiao',
    version='0.0.1',
    install_requires=['Cython>=0.29', 'eigency>=1.77'],
    python_requires='>=3',
    ext_modules=cythonize(ext),
)
