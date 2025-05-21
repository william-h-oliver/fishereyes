# Third-party imports
import jax.numpy as jnp
from matplotlib.collections import PathCollection

# Local imports
from fishereyes.visualization import scatter_colored_by_covariance_shape, covariance_to_color


def test_covariance_to_color(dummy_data):
    """
    Test the covariance_to_color function.
    """
    _, sigma = dummy_data  # Use the covariance matrices from the dummy data

    # Call the function
    colors, color_scale_max = covariance_to_color(sigma)

    print(colors[:, 0].max())
    print(colors[:, 1].max())
    print(colors[:, 2].max())

    # Assert that the output shapes are correct
    assert colors.shape == (sigma.shape[0], 3), f"Expected colors shape {(sigma.shape[0], 3)}, got {colors.shape}"
    assert isinstance(color_scale_max, float), "color_scale_max should be a float"

    # Assert that the colors are within valid RGB range
    assert jnp.all((colors >= 0) & (colors <= 1)), "RGB values should be in the range [0, 1]"


def test_scatter_colored_by_covariance_shape(dummy_data):
    """
    Test the scatter_colored_by_covariance_shape function.
    """
    y, sigma = dummy_data  # Use the data points and covariance matrices from the dummy data

    # Call the function
    path_collection, color_scale_max = scatter_colored_by_covariance_shape(y, sigma)

    # Assert that the returned path collection is valid
    assert isinstance(path_collection, PathCollection), "Expected a PathCollection object"

    # Assert that the color scale max is a float
    assert isinstance(color_scale_max, float), "color_scale_max should be a float"

    # Assert that the scatter plot contains the correct number of points
    assert len(path_collection.get_offsets()) == y.shape[0], "Number of points in scatter plot should match input data"