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

@pytest.fixture
def dummy_config(tmp_path) -> str:
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("""
    model:
      name: neural_ode
      params:
        vector_field:
          name: mlp
          params:
            hidden_dims: [4]
            activation: tanh
        time_dependence: true
        time_length: 1.0
        time_steps: 2
        solver_params:
          atol: 0.0001
          rtol: 0.0001
    optimizer:
      name: adam
      params:
        learning_rate: 0.001
    loss:
      name: ssiKLdiv
      params: {}
    training:
      epochs: 2
      batch_size: 8
    """)
    return str(config_path)


@pytest.fixture
def dummy_invalid_config(tmp_path) -> str:
    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text("""
    model:
      name: invalid_model
      params: {}
    """)
    return str(config_path)