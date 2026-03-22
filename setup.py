"""
Build script for optional pybind11 C++ extension.

Usage:
    pip install pybind11
    python setup.py build_ext --inplace

The C extension is optional — the package works without it via numpy fallback.
"""
import sys
import platform
from setuptools import setup, Extension

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("pybind11 not found. Install with: pip install pybind11")
    print("The C extension is optional — dhl_mm works without it.")
    sys.exit(0)

# Platform-specific compiler flags
if platform.system() == "Windows":
    # MSVC flags
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
elif platform.system() == "Darwin":
    # macOS — libomp may or may not be installed
    extra_compile_args = ["-O3", "-std=c++14"]
    extra_link_args = []
    # Try to enable OpenMP if available (brew install libomp)
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
        pass  # No brew / no libomp — build without OpenMP
else:
    # Linux / other Unix
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

setup(
    name="dhl-mm-csparse",
    version="1.0.0",
    description="C++ sparse bracket extension for DHL-MM",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
