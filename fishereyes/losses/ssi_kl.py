# Standard imports
from typing import Optional, Any

# Third-party imports
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.losses.baseloss import ConfigurableLoss


class SymmetrizedScaleInvariantKL(ConfigurableLoss):
    def __init__(self, alpha = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        model: Any,
        params: Any,
        y0: jax.Array,
        eigvals0: jax.Array,
        eigvecs0: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute the scale-invariant symmetrized KL loss for a batch.
        """
        # Handle the mask
        if mask is None:
            mask = jnp.ones(y0.shape[0], dtype=jnp.float32)
        else:
            mask = mask.astype(jnp.float32)  # convert bool to float for multiplication

        # Calculate the Jacobian of the model with respect to the input y0
        def single_jac(y):
            return jax.jacrev(lambda x: model(x, params=params))(y)

        J = jax.vmap(single_jac)(y0)

        # Calculate the loss
        return self._symmetrized_scale_invariant_KL_loss(J, eigvals0, eigvecs0, mask, self.alpha)

    @staticmethod
    @jax.jit
    def _symmetrized_scale_invariant_KL_loss(
        J: jax.Array,
        eigvals: jax.Array,
        eigvecs: jax.Array,
        mask: jax.Array,
        alpha: float,
    ) -> jax.Array:
        # Transform the Jacobian to the eigenspace of sigma0
        J_tilde = jnp.einsum('nij,njk->nik', J, eigvecs)

        # Forward trace
        row_norms_squared = jnp.sum(J_tilde**2, axis=2)
        sum_trace_C = jnp.sum(mask * jnp.sum(eigvals * row_norms_squared, axis=1))

        # Inverse trace
        J_tilde_inv = jnp.linalg.inv(J_tilde)
        row_norms_inv_squared = jnp.sum(J_tilde_inv**2, axis=2)
        sum_trace_Cinv = jnp.sum(mask * jnp.sum(row_norms_inv_squared / eigvals, axis=1))

        # Number of samples and dimensions
        n = mask.sum()
        d = eigvals.shape[1]

        # Calculate the Symmetrized Kullback-Leibler divergence between the 
        # push-forward covariance and the isotropic covariance
        return 0.5 * (sum_trace_C / alpha + sum_trace_Cinv * alpha) / n - d
    
    def calculate_optimal_alpha(self, eigvals: jax.Array) -> None:
        """
        Calculate the optimal alpha for the loss function.

        Parameters:
        - eigvals: Eigenvalues of the covariance matrix [N, D]
        """
        sum_trace_C = jnp.sum(eigvals)
        sum_trace_Cinv = jnp.sum(1.0 / eigvals)
        self.alpha = jnp.sqrt(sum_trace_C / sum_trace_Cinv)

    def calculate_reference_loss(self, eigvals: jax.Array) -> jax.Array:
        """
        Calculate the reference loss for the loss function. Defined as the loss 
        value when the model transformation is identity.

        Parameters:
        - eigvals: Eigenvalues of the covariance matrix [N, D]

        Returns:
        - Reference loss value
        """
        # Forward and inverse trace
        sum_trace_C = jnp.sum(eigvals)
        sum_trace_Cinv = jnp.sum(1.0 / eigvals)

        # Calculate the reference loss
        n, d = eigvals.shape
        return 0.5 * (sum_trace_C / self.alpha + sum_trace_Cinv * self.alpha) / n - d