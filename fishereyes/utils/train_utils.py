# Third-party imports
from jax import jit, value_and_grad
import optax


def loss_and_grad(model, loss_fn, params, y0_batch, eigvals0_batch, eigvecs0_batch):
    def loss_fn_wrapped(params):
        return loss_fn(model, params, y0_batch, eigvals0_batch, eigvecs0_batch)

    loss_val, grads = value_and_grad(loss_fn_wrapped)(params)
    return loss_val, grads


def update(optimizer, params, opt_state, grads):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state