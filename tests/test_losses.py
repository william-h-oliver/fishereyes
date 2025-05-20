# Standard imports
from typing import Tuple

# Third-party imports
import pytest
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.models.mlp import MLP
from fishereyes.losses.registry import LOSS_REGISTRY


@pytest.mark.parametrize("loss_name", LOSS_REGISTRY.keys())
def test_loss_instantiation(loss_name):
    loss_cls = LOSS_REGISTRY[loss_name]
    loss_fn = loss_cls()
    assert loss_fn is not None, f"Failed to instantiate loss function: {loss_name}"


def test_ssi_kl_loss_computation(
    dummy_data: Tuple[jax.Array, jax.Array],
    key: jax.random.key,
) -> None:
    # Dummy data
    y0, sigma0 = dummy_data
    eigvals0, eigvecs0 = jnp.linalg.eigh(sigma0)

    # Dummy model
    model = MLP(input_dim=y0.shape[-1],
                output_dim=y0.shape[-1],
                hidden_dims=[16],
                key=key)
    params = model.parameters()


    loss_cls = LOSS_REGISTRY["ssiKLdiv"]
    loss_fn = loss_cls()

    # Compute loss
    loss = loss_fn(model, params, y0, eigvals0, eigvecs0)
    assert isinstance(loss, jax.Array) and loss.shape == (), "Loss should be a scalar"
    assert loss >= 0, "Loss should be non-negative"
