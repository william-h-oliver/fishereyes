# Standard imports
from typing import Optional, Union

# Third-party imports
import jax


def create_key(seed: Optional[Union[jax.random.key, int]]) -> jax.random.key:
    """
    Create a JAX random key from an integer seed or a JAX key.
    """
    if isinstance(seed, jax.Array):
        return seed
    elif isinstance(seed, int):
        return jax.random.key(seed)
    else:
        return jax.random.key(0)