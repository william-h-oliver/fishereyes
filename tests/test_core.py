# Standard imports
from typing import Tuple

# Third-party imports
import pytest
import jax

# Local imports
from fishereyes.core import FisherEyes


@pytest.fixture
def dummy_fishereyes(
    dummy_data: Tuple[jax.Array, jax.Array],
    key: jax.random.PRNGKey,
) -> FisherEyes:
    y0, _ = dummy_data
    config_path = "tests/configs/test_config.yaml"
    fishi = FisherEyes.from_config(y0.shape[-1], config_path=config_path, key=key)
    return fishi


def test_fit_runs(
    dummy_fishereyes: FisherEyes,
    dummy_data: Tuple[jax.Array, jax.Array],
    key: jax.random.PRNGKey,
) -> None:
    y0, sigma0 = dummy_data
    dummy_fishereyes.fit(y0, sigma0, key)
    assert dummy_fishereyes.loss_history, "Loss history should not be empty after training"
