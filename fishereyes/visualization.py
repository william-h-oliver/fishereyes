# Standard imports
from typing import Optional, Tuple, Dict, Any

# Third-party imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection

def scatter_colored_by_covariance_shape(
    y: jax.Array,
    sigma: jax.Array,
    color_scale_max: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    scatterKwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[PathCollection, float]:
    """
    Scatter plot of points with colors based on the shape of their covariance.

    Parameters
    ----------
    y: jax.Array
        The data points to be plotted.
    sigma: jax.Array
        The covariance matrix corresponding to the data points.
    saturation_scale: Optional[float]
        The scale for saturation of colors. If None, the maximum saturation is used.
    ax: Optional[plt.Axes]
        The axes on which to plot the data. If None, the current axes are used.
    scatterKwargs: Optional[Dict[str, Any]]
        Additional keyword arguments for the scatter plot.

    Returns
    -------
    pathCollection: plt.collections.PathCollection
        The collection of paths created by the scatter plot.
    saturation_scale: float
        The saturation scale used for the colors.
    """
    # === Check if the axes have been provided ===
    if ax is None: ax = plt.gca()
    
    # === Set default scatter kwargs ===
    if scatterKwargs is None:
        scatterKwargs = {
            "edgecolor": "black",
            "linewidth": 0.5,
            "s": 20,
        }
    else:
        if 'c' in scatterKwargs: del scatterKwargs['c']
        if 'color' in scatterKwargs: del scatterKwargs['color']
        if 'fc' in scatterKwargs: del scatterKwargs['fc']
        if 'facecolor' in scatterKwargs: del scatterKwargs['facecolor']

    # === Calculate colors from covariance matrix ===
    colors, color_scale_max = covariance_to_color(sigma, color_scale_max)

    # === Convert to numpy and reduce to 2D for plotting ===
    pathCollection = ax.scatter(*y.__array__().T, facecolor=colors, **scatterKwargs)

    return pathCollection, color_scale_max

def covariance_to_color(
    sigma: jax.Array,
    color_scale_max: Optional[float] = None,
) -> Tuple['numpy.ndarray', float]:
    """
    Convert covariance matrix to RGB color representation.

    Parameters
    ----------
    sigma: jax.Array
        The covariance matrix to be converted.
    color_scale_max: Optional[float]
        The maximum value for color scaling. If None, the maximum eigenvalue is used.
    
    Returns
    -------
    rgb: jax.Array
        The RGB color representation of the covariance matrix.
    color_scale_max: float
        The maximum value used for color scaling.
    """
    # === Convert covariance matrix to RGB color ===
    eigvals, eigvecs = jnp.linalg.eigh(sigma) # Eigendecomposition

    # === Calculate hues ===
    angles = jnp.arctan2(eigvecs[:, 1, -1], eigvecs[:, 0, -1]) # Angle of the last eigenvector
    hues = ((2 * angles) % (2 * jnp.pi)) / (2 * jnp.pi) # in [0, 1] with opposite directions mapping to the same color

    # === Calculate saturations ===
    ratios = jnp.clip(eigvals[:, -1] / eigvals[:, 0], 1.0, None)
    saturations = jnp.log(ratios) # Rescale for easier visualization of large ratios
    if color_scale_max is None: # Use max eigenvalue for color scaling if not provided
        color_scale_max = jnp.max(saturations).item()
    if color_scale_max == 0: # Avoid division by zero
        color_scale_max = 1.0
    saturations = jnp.clip(saturations / color_scale_max, 0.0, 1.0) # Saturation in [0, 1]
    
    # === Set values ===
    values = jnp.ones_like(hues) # Value fixed to 1

    # === Convert HSV to RGB ===
    hsv = jnp.column_stack([hues, saturations, values])
    rgb = mcolors.hsv_to_rgb(hsv.__array__())
    return rgb, color_scale_max