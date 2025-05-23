# Standard imports
from typing import Tuple, Any

# Third-party imports
import jax
import optax


def loss_and_grad(
    model: Any,
    loss_fn: Any,
    params: Any,
    y0: jax.Array,
    eigvals0: jax.Array,
    eigvecs0: jax.Array
) -> Tuple[float, jax.Array]:
    """
    Compute the loss and gradients for a batch of data.

    Parameters
    ----------
    model: Any
        The model to be trained.
    loss_fn: Any
        The loss function to be used for training.
    params: Any
        The parameters of the model.
    y0: jax.Array
        The input data for the model.
    eigvals0: jax.Array
        The eigenvalues of the covariance matrix.
    eigvecs0: jax.Array
        The eigenvectors of the covariance matrix.
    
    Returns
    -------
    loss_val: float
        The computed loss value.
    grads: jax.Array
        The computed gradients of the loss with respect to the model parameters.
    """
    # Define a wrapped loss function to compute gradients
    def loss_fn_wrapped(params):
        return loss_fn(model, params, y0, eigvals0, eigvecs0)

    # Compute the loss and gradients
    loss_val, grads = jax.value_and_grad(loss_fn_wrapped)(params)
    
    return loss_val, grads

def update(
    optimizer: Any,
    params: Any,
    opt_state: Any,
    grads: jax.Array,
) -> Tuple[Any, Any]:
    """
    Update the model parameters using the optimizer.

    Parameters
    ----------
    optimizer: Any
        The optimizer to be used for updating the parameters.
    params: Any
        The current parameters of the model.
    opt_state: Any
        The current state of the optimizer.
    grads: jax.Array
        The gradients of the loss with respect to the model parameters.

    Returns
    -------
    new_params: Any
        The updated parameters of the model.
    new_opt_state: Any
        The updated state of the optimizer.
    """
    # Apply the optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # Apply the updates to the parameters
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state