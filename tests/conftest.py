import pytest
import jax
import jax.numpy as jnp
from jax.random import PRNGKey


@pytest.fixture(scope="module")
def key():
    return PRNGKey(0)


@pytest.fixture
def dummy_data():
    N, D = 10, 3
    y0 = jnp.linspace(-1.0, 1.0, N * D).reshape(N, D)
    sigma0 = jnp.stack([jnp.eye(D) * (1 + 0.1 * i) for i in range(N)])
    return y0, sigma0