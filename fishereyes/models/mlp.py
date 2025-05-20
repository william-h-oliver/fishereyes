# Standard imports
from typing import List, Optional, Union, Dict, Any

# Third-party imports
import jax
import jax.numpy as jnp

# Local imports
from fishereyes.models.basemodel import ConfigurableModel


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
        if isinstance(key, jax.Array):
            self.key = key
        elif isinstance(key, int):
            key = jax.random.key(key)
            self.key = key
        else:
            key = jax.random.key(0)
            self.key = key

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
    
    def as_config(self) -> Dict[str, Any]:
        return {
            "name": "MLP",
            "params": {
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
            }
        }

    def __call__(
        self,
        x: jax.Array,
        params: Optional[Dict[str, Any]] = None
    ) -> jax.Array:
        params = self._params if params is None else params

        for i, layer in enumerate(params['layers']):
            x = jnp.dot(x, layer['w']) + layer['b']
            if i < len(params['layers']) - 1:
                x = self.activation_fn(x)
        return x

    def parameters(self) -> Dict[str, Any]:
        return self._params

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self._params = params

