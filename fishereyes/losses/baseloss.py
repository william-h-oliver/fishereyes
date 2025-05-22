# Standard imports
from abc import ABC, abstractmethod


# Third-party imports
import jax

class ConfigurableLoss(ABC):
    """
    Abstract base class for losses.
    """
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_optimal_alpha(self, eigvals: jax.Array) -> None:
        """
        Calculate the optimal alpha for the loss function.

        Parameters:
        - eigvals: Eigenvalues of the covariance matrix [N, D]
        """
        pass

    @abstractmethod
    def calculate_reference_loss(self, *args, **kwargs) -> jax.Array:
        """
        Calculate the reference loss for the loss function. Defined as the loss 
        value when the model transformation is identity.

        Parameters:
        - args: Positional arguments
        - kwargs: Keyword arguments
        """
        pass