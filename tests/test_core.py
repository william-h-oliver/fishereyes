import jax
import jax.numpy as jnp
import pytest

from fishereyes.core import FisherEyes
from fishereyes.models.mlp import MLP
from fishereyes.losses.ssi_kl import SymmetrizedScaleInvariantKL
from fishereyes.optimizers.registry import OPTIMIZER_REGISTRY


@pytest.fixture
def dummy_fishereyes(dummy_data, key):
    y0, sigma0 = dummy_data
    model = MLP(input_dim=y0.shape[1], output_dim=y0.shape[1], hidden_dims=[16])
    optimizer = OPTIMIZER_REGISTRY["adam"](learning_rate=1e-2)
    opt_state = optimizer.init(model.parameters())
    loss_fn = SymmetrizedScaleInvariantKL()
    return FisherEyes(model, optimizer, opt_state, loss_fn, epochs=1, batch_size=5)


def test_fit_runs(dummy_fishereyes, dummy_data, key):
    y0, sigma0 = dummy_data
    dummy_fishereyes.fit(y0, sigma0, key)
    assert dummy_fishereyes.loss_history, "Loss history should not be empty after training"
