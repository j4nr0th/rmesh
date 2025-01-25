### RMESH
`rmesh` is a module which can perform mesh generation for 2D, rectangular, elliptical, semi-structured meshes. It does that by iteratively solving the Laplace equation for the position of the mesh nodes.

## Dependencies
The module depends on [numpy](https://numpy.org) for its arrays, which are used to access most of the mesh data.

For building the actual package there are additional dependencies on [CMake](https://cmake.org/) and [scikit-build-core](https://github.com/scikit-build/scikit-build), which are needed to build the C extension which is used by the module to quickly generate and solve the system of equations.

## Building and Installation
To build `rmesh`, [numpy](https://numpy.org), [CMake](https://cmake.org/), [scikit-build-core](https://github.com/scikit-build/scikit-build) are needed. To build and install the Python package follow the following steps:

1. Clone the repository via `git clone <repo> <dir> && cd <dir>`
2. Pull dependencies from git `git submodule init`. This will pull the library used for linear algebra and iterative solver.
3. Build the package wheel(s) using Python `python -m build <output-dir>`
4. Install the package from the wheel with `pip install <output-dir>/rmesh-*.whl`

## Documentation

Built documentation is hosted using Github pages [here](https://j4nr0th.github.io/rmesh/). If you wish to build them yourself, you can do
so by installing the package with optional dependency `[doc]` (for example, if you are in the source directory, call `pip install .[doc]`).
Following that, you can build the documentation pages by calling `sphinx-build doc <out-dir>`.
