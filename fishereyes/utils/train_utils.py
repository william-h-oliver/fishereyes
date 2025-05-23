# Standard imports
from typing import Union, List, Tuple

# Third-party imports
import jax
import jax.numpy as jnp
import optax


def shuffle_and_split_batches(
    n: int,
    batch_size: Union[int, None],
    key: jax.random.key,
) -> Tuple[List[jax.Array], jax.Array]:
    """
    Shuffle the data and split it into batches.
    """
    if batch_size is None:
        # No batching, return the entire dataset as a single batch
        batches = [jnp.arange(n)]
    else:
        # Shuffle the data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)

        # Adjust number of batches to include all data
        last_batch_adjust = n % batch_size if batch_size <= n else 0
        if last_batch_adjust != 0: last_batch_adjust = batch_size

        # Split into batches
        batches = [perm[i:i + batch_size] for i in range(0, n + last_batch_adjust, batch_size)]

    return batches, key

def loss_and_grad(model, loss_fn, params, y0, eigvals0, eigvecs0):
    """
    Compute the loss and gradients for a batch of data.
    """
    # Define a wrapped loss function to compute gradients
    def loss_fn_wrapped(params):
        return loss_fn(model, params, y0, eigvals0, eigvecs0)

    # Compute the loss and gradients
    loss_val, grads = jax.value_and_grad(loss_fn_wrapped)(params)
    
    return loss_val, grads

def update(optimizer, params, opt_state, grads):
    """
    Update the model parameters and optimizer state.
    """
    # Apply the optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # Apply the updates to the parameters
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state