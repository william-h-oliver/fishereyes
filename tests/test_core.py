# Standard imports
from typing import Tuple

# Third-party imports
import pytest
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.core import FisherEyes


@pytest.fixture
def dummy_fishereyes(
    dummy_data: Tuple[jax.Array, jax.Array],
    dummy_config: str,
    key: jax.random.key,
) -> FisherEyes:
    y0, _ = dummy_data
    fishi = FisherEyes.from_config(y0.shape[-1], config_path=dummy_config, key=key)
    return fishi


def test_init_fishereyes(
    dummy_fishereyes: FisherEyes,
) -> None:
    assert isinstance(dummy_fishereyes, FisherEyes), "FisherEyes instance not created correctly"
    assert dummy_fishereyes.model is not None, "Model should be initialized"
    assert dummy_fishereyes.optimizer is not None, "Optimizer should be initialized"
    assert dummy_fishereyes.opt_state is not None, "Optimizer state should be initialized"
    assert dummy_fishereyes.loss_fn is not None, "Loss function should be initialized"
    assert dummy_fishereyes.epochs > 0, "Epochs should be greater than 0"
    assert dummy_fishereyes.config is not None, "Config should be initialized"
    assert dummy_fishereyes.loss_history == [], "Loss history should be empty at initialization"


def test_from_config_invalid(
    dummy_data: Tuple[jax.Array, jax.Array],
    dummy_invalid_config: str,
    key: jax.random.key,
) -> None:
    y0, _ = dummy_data
    with pytest.raises(KeyError):
        FisherEyes.from_config(y0.shape[-1], config_path=dummy_invalid_config, key=key)


def test_fit_runs(
    dummy_fishereyes: FisherEyes,
    dummy_data: Tuple[jax.Array, jax.Array],
    key: jax.random.key,
) -> None:
    y0, sigma0 = dummy_data
    dummy_fishereyes.fit(y0, sigma0, key)
    assert dummy_fishereyes.loss_history, "Loss history should not be empty after training"


def test_fit_invalid_inputs(
    dummy_fishereyes: FisherEyes,
    key: jax.random.key,
) -> None:
    y0 = jnp.array([[1.0, 2.0]])
    sigma0 = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])  # Mismatched shapes
    with pytest.raises(ValueError):
        dummy_fishereyes.fit(y0, sigma0, key)


def test_predict(
    dummy_fishereyes: FisherEyes,
    dummy_data: Tuple[jax.Array, jax.Array],
) -> None:
    """
    Test the predict function of the FisherEyes class.
    """
    y0, sigma0 = dummy_data

    # Call the predict function
    y1, sigma1 = dummy_fishereyes.predict(y0, sigma0)

    # Assert that the output shapes are correct
    assert y1.shape == y0.shape, f"Expected y1 shape {y0.shape}, got {y1.shape}"
    assert sigma1.shape == sigma0.shape, f"Expected sigma1 shape {sigma0.shape}, got {sigma1.shape}"

    # Assert that the transformed covariance matrices are symmetric
    assert jnp.allclose(sigma1, jnp.swapaxes(sigma1, -1, -2)), "Transformed covariance matrices should be symmetric"

    # Assert that the transformed covariance matrices are positive semi-definite
    eigvals = jnp.linalg.eigvalsh(sigma1)
    assert jnp.all(eigvals >= 0), "Transformed covariance matrices should be positive semi-definite"
