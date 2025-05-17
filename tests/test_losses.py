import jax.numpy as jnp
from fishereyes.losses.ssi_kl import SymmetrizedScaleInvariantKL
from fishereyes.models.mlp import MLP


def test_ssi_kl_loss_runs(dummy_data):
    y0, sigma0 = dummy_data
    model = MLP(input_dim=3, output_dim=3, hidden_dims=[8])
    params = model.parameters()

    eigvals, eigvecs = jnp.linalg.eigh(sigma0)
    loss_fn = SymmetrizedScaleInvariantKL()
    loss = loss_fn(model, params, y0, eigvals, eigvecs)
    assert isinstance(loss, jnp.ndarray) and loss.shape == (), "Loss should be a scalar"
