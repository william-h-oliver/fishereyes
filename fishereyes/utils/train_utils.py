# Standard imports
from typing import Tuple

# Third-party imports
import jax
import jax.numpy as jnp
import optax


def shuffle_and_split_batches(
    batch_size: int,
    y0: jax.Array,
    eigvals0: jax.Array,
    eigvecs0: jax.Array,
    key: jax.Array,
) -> Tuple[
    Tuple[
        jax.Array,  # y0_batches
        jax.Array,  # eigvals0_batches
        jax.Array,  # eigvecs0_batches
        jax.Array,  # mask_batches
    ],
    jax.Array,  # key
]:
    """
    Shuffle the data and split it into padded batches with masks.

    Returns:
    - y0_batches: List of [batch_size, D]
    - eigvals0_batches: List of [batch_size, D]
    - eigvecs0_batches: List of [batch_size, D, D]
    - mask_batches: List of [batch_size] boolean masks (True = real data, False = padded)
    - key: updated PRNG key
    """
    n, d = y0.shape
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n)

    # Compute padding
    pad_size = batch_size - (n % batch_size) if n % batch_size != 0 else 0

    # Pad y0 with zeros
    y0 = jnp.pad(y0[perm], ((0, pad_size), (0, 0)))

    # Pad eigvals0 with ones (identity eigenvalues)
    eigvals_identity = jnp.ones((pad_size, d))
    eigvals0 = jnp.concatenate([eigvals0[perm], eigvals_identity], axis=0)

    # Pad eigvecs0 with identity matrices (identity eigenvectors)
    eigvecs_identity = jnp.tile(jnp.eye(d)[None, :, :], (pad_size, 1, 1))
    eigvecs0 = jnp.concatenate([eigvecs0[perm], eigvecs_identity], axis=0)

    # Mask: 1 for real data, 0 for padding
    mask = jnp.concatenate([jnp.ones(n, dtype=jnp.float32), jnp.zeros(pad_size, dtype=jnp.float32)])

    # Split into batches
    num_batches = y0.shape[0] // batch_size
    y0_batches = jnp.stack(jnp.split(y0, num_batches))
    eigvals0_batches = jnp.stack(jnp.split(eigvals0, num_batches))
    eigvecs0_batches = jnp.stack(jnp.split(eigvecs0, num_batches))
    mask_batches = jnp.stack(jnp.split(mask, num_batches))

    return (y0_batches, eigvals0_batches, eigvecs0_batches, mask_batches), key


def loss_and_grad(model, loss_fn, params, y0_batch, eigvals0_batch, eigvecs0_batch, mask_batch):
    """
    Compute the loss and gradients for a batch of data.
    """
    # Define a wrapped loss function to compute gradients
    def loss_fn_wrapped(params):
        return loss_fn(model, params, y0_batch, eigvals0_batch, eigvecs0_batch, mask_batch)

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