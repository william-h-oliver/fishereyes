# Standard imports
from typing import Tuple

# Third-party imports
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.losses.ssi_kl import SymmetrizedScaleInvariantKL
from fishereyes.models.mlp import MLP


def test_ssi_kl_loss_runs(
    dummy_data: Tuple[jax.Array, jax.Array],
    key: jax.random.key,
) -> None:
    y0, sigma0 = dummy_data
    model = MLP(input_dim=y0.shape[-1],
                output_dim=y0.shape[-1],
                hidden_dims=[16],
                key=key)
    params = model.parameters()

    eigvals, eigvecs = jnp.linalg.eigh(sigma0)
    loss_fn = SymmetrizedScaleInvariantKL()
    loss = loss_fn(model, params, y0, eigvals, eigvecs)
    assert isinstance(loss, jax.Array) and loss.shape == (), "Loss should be a scalar"
