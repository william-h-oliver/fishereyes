import jax
import jax.numpy as jnp
from typing import List, Optional, Dict

from fishereyes.models.basemodel import ConfigurableModel


class MLP(ConfigurableModel):
    def __init__(self, hidden_dims: List[int], activation: str = "tanh"):
        self.hidden_dims = hidden_dims
        self.activation_fn = getattr(jnp, activation)

        # Placeholder for parameters (should be set externally or through init)
        self._params = None

    def init_parameters(self, input_dim, key):
        """Initialize parameters given input dim and PRNG key."""
        layers = []
        dims = [input_dim] + self.hidden_dims + [input_dim]

        for i in range(len(dims) - 1):
            k1, key = jax.random.split(key)
            w = jax.random.normal(k1, (dims[i], dims[i + 1])) * 0.01
            b = jnp.zeros((dims[i + 1],))
            layers.append({'w': w, 'b': b})

        self._params = {'layers': layers}
        return self._params

    def __call__(self, x, params=None):
        if params is None:
            params = self._params

        for i, layer in enumerate(params['layers']):
            x = jnp.dot(x, layer['w']) + layer['b']
            if i < len(params['layers']) - 1:
                x = self.activation_fn(x)
        return x

    def parameters(self):
        return self._params

    def set_parameters(self, params):
        self._params = params

