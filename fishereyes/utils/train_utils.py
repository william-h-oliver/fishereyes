# Standard imports
from typing import List, Tuple

# Third-party imports
import jax
import optax


def shuffle_and_split_batches(
    batch_size: int,
    y0: jax.Array,
    eigvals0: jax.Array,
    eigvecs0: jax.Array,
    key: jax.random.key,
) -> Tuple[List[jax.Array], List[jax.Array], List[jax.Array], jax.random.key]:
    """
    Shuffle the data and split it into batches.
    """
    # Shuffle the data
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, y0.shape[0])

    # Adjust number of batches to include all data
    last_batch_adjust = y0.shape[0] % batch_size if batch_size <= y0.shape[0] else 0
    if last_batch_adjust != 0: last_batch_adjust = batch_size

    # Split into batches
    batches = [perm[i:i + batch_size] for i in range(0, y0.shape[0] + last_batch_adjust, batch_size)]
    y0_batches = [y0[batch] for batch in batches]
    eigvals0_batches = [eigvals0[batch] for batch in batches]
    eigvecs0_batches = [eigvecs0[batch] for batch in batches]

    return y0_batches, eigvals0_batches, eigvecs0_batches, key

def loss_and_grad(model, loss_fn, params, y0_batch, eigvals0_batch, eigvecs0_batch):
    """
    Compute the loss and gradients for a batch of data.
    """
    # Define a wrapped loss function to compute gradients
    def loss_fn_wrapped(params):
        return loss_fn(model, params, y0_batch, eigvals0_batch, eigvecs0_batch)

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