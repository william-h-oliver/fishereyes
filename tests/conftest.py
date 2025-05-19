import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def dummy_data():
    N, D = 16, 2
    y0 = jnp.linspace(-1.0, 1.0, N * D).reshape(N, D)
    sigma0 = jnp.stack([jnp.eye(D) * (1 + 0.1 * i) for i in range(N)])
    return y0, sigma0