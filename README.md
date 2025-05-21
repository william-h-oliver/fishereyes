# Welcome to FisherEyes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/fishereyes/ci.yml?branch=main)](https://github.com/william-h-oliver/fishereyes/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fishereyes/badge/)](https://fishereyes.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/fishereyes/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/fishereyes)

The fishereyes package provides tools to learn smooth, invertible transformations of data where each point represents a measurement and is associated with its own uncertainty in the form of a covariance matrix. It enables the transformation of locally anisotropic data into a space where push-forward uncertainties are isotropic and uniform. The core FisherEyes class offers a modular interface to plug in different transformation models, loss functions, and optimizers -- supporting uncertainty-aware learning in a wide range of scientific and machine learning tasks.

## Installation

The Python package `fishereyes` can be installed from PyPI:

```
python -m pip install fishereyes
```

## Basic usage

FisherEyes can easily be applied to any point-based data set with accompanying covariance matrices so long as these are expressed as `jax.Array` types with shapes `(n_samples, d_features)` and `(n_samples, d_features, d_features)` respectively.

So given some data...

```python
import jax
import jax.numpy as jnp

# Set seed for reproducibility
key = jax.random.key(0)
subkey, key = jax.random.split(key)

# Generate some uncertain data
n, d = 512, 2
y0 = jax.random.normal(subkey, (n, d)) # Gaussian blob
sigma0 = jnp.eye(d) + jnp.einsum('ni,nj->nij', y0, y0) # Radial-dependent covariance
```

... we can initialize the FisherEyes class according to the dimensionality of the data and to the settings in the `default_config.yaml` file...

```python
from fishereyes import FisherEyes

fish = FisherEyes.from_config(data_dim=y0.shape[-1], config_path=None, key=key)
fish.fit(y0, sigma0)
```

... and that's it, FisherEyes has found a diffeomorphic transformation of the data such that the push-forward covariance matrices are isotropic and homoskedastic!

### Visualising the estimated density field of the input data

For low-dimensional input data, like we have in this example, it is then possible to visualize the
uncertain data's transformation.

```python
import matplotlib.pyplot as plt
from fishereyes import visualization

# Make a two panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

# Plot the original data (saving that colour scale for better comparison)
_, color_scale_max = visualization.scatter_colored_by_covariance_shape(y0, sigma0, ax=ax1)

# Plot the transformed data
y1, sigma1 = fish.predict(y0, sigma0)
visualization.scatter_colored_by_covariance_shape(y1, sigma1, color_scale_max=color_scale_max, ax=ax2)

# Tidy up
ax1.set_title('Original data')
ax1.set_xlabel('y[0]')
ax1.set_ylabel('y[1]')
ax1.set_aspect('equal')
ax1.set_title('Transformed data')
ax1.set_xlabel('f(y)[0]')
ax1.set_ylabel('f(y)[1]')
ax1.set_aspect('equal')

# Show plot
plt.show()
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
