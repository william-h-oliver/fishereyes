import jax
import jax.numpy as jnp

from fishereyes.models.mlp import MLP


def test_mlp_forward_shape():
    model = MLP(input_dim=3, output_dim=2, hidden_dims=[4])
    x = jnp.ones((5, 3))
    y = model(x, model.parameters())
    assert y.shape == (5, 2), f"Expected shape (5, 2), got {y.shape}"
