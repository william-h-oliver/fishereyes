# Welcome to FisherEyes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/fishereyes/ci.yml?branch=main)](https://github.com/william-h-oliver/fishereyes/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fishereyes/badge/)](https://fishereyes.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/fishereyes/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/fishereyes)

## Installation

The Python package `fishereyes` can be installed from PyPI:

```
python -m pip install fishereyes
```

## Development installation

If you want to contribute to the development of `fishereyes`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/william-h-oliver/fishereyes
cd fishereyes
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
