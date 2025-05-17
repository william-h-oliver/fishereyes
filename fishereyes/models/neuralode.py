import jax.numpy as jnp
from jax.experimental.ode import odeint
from typing import Optional

from fishereyes.models.basemodel import ConfigurableModel


class NeuralODE(ConfigurableModel):
    def __init__(self, vector_field, time_length=1.0, time_steps=10,
                 ode_solver="odeint", solver_params=None):
        self.vector_field = vector_field
        self.time_length = time_length
        self.time_steps = time_steps
        self.ode_solver = ode_solver
        self.solver_params = solver_params or {}

        self.ts = jnp.linspace(0.0, time_length, time_steps)

    def init_parameters(self, input_dim, key):
        return {"vector_field": self.vector_field.init_parameters(input_dim + 1, key)}

    def __call__(self, y0, ts=None, params=None):
        if params is None:
            params = self.parameters()
        ts = self.ts if ts is None else ts
        return odeint(self._wrapped_vector_field, y0, ts, params)

    def _wrapped_vector_field(self, y, t, params):
        # Ensure t is broadcasted to match batch dimension
        t_input = jnp.full((y.shape[0], 1), t) if y.ndim > 1 else jnp.array([t])
        input_with_time = jnp.concatenate([y, t_input], axis=-1)
        return self.vector_field(input_with_time, params=params["vector_field"])
    
    def parameters(self):
        return {
            "vector_field": self.vector_field.parameters()
        }

    def set_parameters(self, params):
        self.vector_field.set_parameters(params["vector_field"])

