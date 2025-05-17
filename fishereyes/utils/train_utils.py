# Third-party imports
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax

@jit
def loss_and_grad(model, loss_fn, params, y0_batch, sigma0_batch):
    def loss_fn_wrapped(params):
        yT_batch = model.apply(params, y0_batch)
        return loss_fn(y0_batch, yT_batch, sigma0_batch)

    loss_val, grads = value_and_grad(loss_fn_wrapped)(params)
    return loss_val, grads

@jit
def update(optimizer, params, opt_state, grads):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state