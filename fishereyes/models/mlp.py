# Standard imports
from functools import partial
from typing import List, Optional, Union, Dict, Any

# Third-party imports
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.models.basemodel import ConfigurableModel
from fishereyes.utils.key_utils import create_key


class MLP(ConfigurableModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        key: Optional[Union[jax.random.key, int]] = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.key = create_key(key)

        self.activation_fn = getattr(jnp, activation)

        # Initialize parameters given dims and PRNG key
        layers = []
        dims = [input_dim] + self.hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            subkey, key = jax.random.split(key)
            w = jax.random.normal(subkey, (dims[i], dims[i + 1])) * 0.01
            b = jnp.zeros((dims[i + 1],))
            layers.append({'w': w, 'b': b})

        self._params = {'layers': layers}

    def __call__(
        self,
        x: jax.Array,
        params: Optional[Dict[str, Any]] = None
    ) -> jax.Array:
        params = self._params if params is None else params

        return self._forward(x, self.activation_fn, params)
    
    @staticmethod
    @partial(jax.jit, static_argnames=["activation_fn"])
    def _forward(
        x: jax.Array,
        activation_fn: Any,
        params: Dict[str, Any]
    ) -> jax.Array:
        # Do forward pass through the MLP
        for layer in params['layers']:
            x = activation_fn(jnp.dot(x, layer['w']) + layer['b'])
        return x

    def parameters(self) -> Dict[str, Any]:
        return self._params

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self._params = params

