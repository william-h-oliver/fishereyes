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
        J: jax.Array,
        eigvals0: jax.Array,
        eigvecs0: jax.Array,
    ) -> jax.Array:
        """
        Compute the scale-invariant symmetrized KL loss for a batch.

        Parameters:
        - J: Jacobian of the model output w.r.t. inputs [N, D, D]
        - eigvals0: Eigenvalues of the covariance matrix [N, D]
        - eigvecs0: Eigenvectors of the covariance matrix [N, D, D]

        Returns:
        - scalar loss value
        """
        # Compute loss
        return self._symmetrized_scale_invariant_KL_loss(J, eigvals0, eigvecs0, self.alpha)

    @staticmethod
    @jax.jit
    def _symmetrized_scale_invariant_KL_loss(
        J: jax.Array,
        eigvals: jax.Array,
        eigvecs: jax.Array,
        alpha: float,
    ) -> jax.Array:
        # Transform the Jacobian to the eigenspace of sigma0
        J_tilde = jnp.einsum('nij,njk->nik', J, eigvecs)

        # Forward trace
        row_norms_squared = jnp.sum(J_tilde**2, axis=2)
        sum_trace_C = jnp.sum(eigvals * row_norms_squared)

        # Inverse trace
        J_tilde_inv = jnp.linalg.inv(J_tilde)
        row_norms_inv_squared = jnp.sum(J_tilde_inv**2, axis=2)
        sum_trace_Cinv = jnp.sum(row_norms_inv_squared / eigvals)

        n, d = eigvals.shape
        return 0.5 * (sum_trace_C / alpha + sum_trace_Cinv * alpha) / n - d
    
    def calculate_optimal_alpha(self, eigvals: jax.Array) -> None:
        """
        Calculate the optimal alpha for the loss function. Defined as the scalar 
        value that minimizes the loss when the model transformation is identity.

        Parameters:
        - eigvals: Eigenvalues of the covariance matrix [N, D]
        """
        # Forward and inverse trace
        sum_trace_C = jnp.sum(eigvals)
        sum_trace_Cinv = jnp.sum(1.0 / eigvals)

        # Calculate the optimal alpha that minimizes the loss given null transformation
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