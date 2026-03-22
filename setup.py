"""
Build script for optional pybind11 C++ extension.

Usage:
    pip install pybind11
    python setup.py build_ext --inplace

The C extension is optional — the package works without it via numpy fallback.
This file is only used for building the C extension, not for the main package
(which uses pyproject.toml).
"""
import sys
import platform
from setuptools import setup, Extension

ext_modules = []

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    # Platform-specific compiler flags
    if platform.system() == "Windows":
        extra_compile_args = ["/O2", "/openmp"]
        extra_link_args = []
    elif platform.system() == "Darwin":
        extra_compile_args = ["-O3", "-std=c++14"]
        extra_link_args = []
        import subprocess
        try:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                omp_prefix = result.stdout.strip()
                extra_compile_args += ["-Xpreprocessor", "-fopenmp",
                                       f"-I{omp_prefix}/include"]
                extra_link_args += [f"-L{omp_prefix}/lib", "-lomp"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    else:
        extra_compile_args = ["-O3", "-fopenmp", "-std=c++14"]
        extra_link_args = ["-fopenmp"]

    ext_modules = [
        Pybind11Extension(
            "dhl_mm._csparse",
            ["dhl_mm/csrc/sparse_bracket.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
