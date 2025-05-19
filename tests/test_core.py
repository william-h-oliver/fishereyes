import jax
import jax.numpy as jnp
import pytest

from fishereyes.core import FisherEyes


@pytest.fixture
def dummy_fishereyes(dummy_data, key):
    y0, _ = dummy_data
    config_path = "tests/configs/test_config.yaml"
    fishi = FisherEyes.from_config(y0.shape[-1], config_path=config_path, key=key)
    return fishi



def test_fit_runs(dummy_fishereyes, dummy_data, key):
    y0, sigma0 = dummy_data
    dummy_fishereyes.fit(y0, sigma0, key)
    assert dummy_fishereyes.loss_history, "Loss history should not be empty after training"
