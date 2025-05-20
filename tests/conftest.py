# Standard imports
from typing import Tuple

# Third-party imports
import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(scope="module")
def key() -> jax.random.key:
    return jax.random.key(0)


@pytest.fixture
def dummy_data() -> Tuple[jax.Array, jax.Array]:
    N, D = 16, 2
    y0 = jnp.linspace(-1.0, 1.0, N * D).reshape(N, D)
    sigma0 = jnp.stack([jnp.eye(D) * (1 + 0.1 * i) for i in range(N)])
    return y0, sigma0