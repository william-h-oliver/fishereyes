import jax
import jax.numpy as jnp
from typing import Sequence

class MLP:
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: str = "tanh", key=None):
        if key is None:
            key = jax.random.PRNGKey(0)

        # Activation function
        activations = {
            "tanh": jnp.tanh,
            "relu": jax.nn.relu,
            "gelu": jax.nn.gelu,
        }
        self.activation = activations[activation]

        self.shapes = [(input_dim if i == 0 else hidden_dims[i - 1], dim) for i, dim in enumerate(hidden_dims)]
        self.shapes.append((hidden_dims[-1], input_dim))  # output dim same as input

        self.params = []
        for i, (in_dim, out_dim) in enumerate(self.shapes):
            k1, key = jax.random.split(key)
            w = jax.random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
            b = jnp.zeros((out_dim,))
            self.params.append({'w': w, 'b': b})

    def __call__(self, x, params=None):
        if params is None:
            params = self.params
        for i, layer in enumerate(params):
            x = jnp.dot(x, layer['w']) + layer['b']
            if i < len(params) - 1:
                x = self.activation(x)
        return x

    def parameters(self):
        return self.params

    def set_parameters(self, params):
        self.params = params
