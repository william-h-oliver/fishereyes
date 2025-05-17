import jax.numpy as jnp
import jax
from fishereyes.optimizers.registry import OPTIMIZER_REGISTRY


def test_adam_optimizer_update():
    optimizer = OPTIMIZER_REGISTRY["adam"](learning_rate=1e-2)
    params = {"w": jnp.ones((3, 3))}
    grads = {"w": jnp.full((3, 3), 0.5)}
    opt_state = optimizer.init(params)

    updates, new_state = optimizer.update(grads, opt_state, params)
    assert isinstance(updates, dict), "Updates should be a dictionary"
    assert updates["w"].shape == (3, 3), "Update shape should match parameter shape"
