# Standard imports
from typing import Dict, Any

# Third-party imports
import jax
import jax.numpy as jnp


class SymmetrizedScaleInvariantKL:
    def __init__(self):
        # No hyperparameters for now, but leave __init__ for extensibility
        pass

    def as_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the loss function configuration.
        This is useful for saving/loading configurations.
        """
        return {
            "name": "SymmetrizedScaleInvariantKL",
            "params": {}
        }

    def __call__(
        self,
        model: Any,
        params: Any,
        y0: jax.Array,
        eigvals0: jax.Array,
        eigvecs0: jax.Array,
    ) -> jax.Array:
        """
        Compute the scale-invariant symmetrized KL loss for a batch.

        Parameters:
        - model: A callable model (e.g. MLP or NeuralODE)
        - params: Model parameters
        - y0: Input data [N, D]
        - eigvals0: Eigenvalues of the covariance matrix [N, D]
        - eigvecs0: Eigenvectors of the covariance matrix [N, D, D]

        Returns:
        - scalar loss value
        """

        # Compute Jacobians of model output w.r.t. inputs for each sample
        def single_jac(y):
            return jax.jacrev(lambda x: model(x, params=params))(y)  # shape (D, D)

        J = jax.vmap(single_jac)(y0)  # shape (N, D, D)

        return self._symmetrized_scale_invariant_KL_loss(J, eigvals0, eigvecs0)

    @staticmethod
    @jax.jit
    def _symmetrized_scale_invariant_KL_loss(
        J: jax.Array,
        eigvals: jax.Array,
        eigvecs: jax.Array,
    ) -> jax.Array:
        # Transform the Jacobian to the eigenspace of sigma0
        J_tilde = jnp.einsum('nij,njk->nik', J, eigvecs)

        # Forward trace
        row_norms_squared = jnp.sum(J_tilde**2, axis=2)
        sum_trace_C = jnp.sum(eigvals * row_norms_squared)

        # Inverse trace
        J_tilde_inv = jnp.linalg.inv(J_tilde)
        # J_tilde_inv = jnp.linalg.solve(J_tilde, jnp.eye(J.shape[-1])) # Safer?
        row_norms_inv_squared = jnp.sum(J_tilde_inv**2, axis=2)
        sum_trace_Cinv = jnp.sum(row_norms_inv_squared / eigvals)

        n, d = eigvals.shape
        return 0.5 * (jnp.sqrt(sum_trace_C * sum_trace_Cinv) / n - d)