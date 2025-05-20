# Third-party imports
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.models.mlp import MLP


def test_mlp_forward_shape(key: jax.random.PRNGKey) -> None:
    model = MLP(input_dim=3, output_dim=2, hidden_dims=[4], key=key)
    x = jnp.ones((5, 3))
    y = model(x)
    assert y.shape == (5, 2), f"Expected shape (5, 2), got {y.shape}"
