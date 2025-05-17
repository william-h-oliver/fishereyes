import jax.numpy as jnp
from jax.experimental.ode import odeint
from typing import Any

class NeuralODE:
    def __init__(self, vector_field, time_length=1.0, time_steps=10, ode_solver="odeint", solver_params=None):
        self.vector_field = vector_field
        self.time_length = time_length
        self.time_steps = time_steps
        self.ode_solver = ode_solver
        self.solver_params = solver_params or {}

        self.ts = jnp.linspace(0.0, time_length, time_steps)

    def __call__(self, y0, sigma0=None, params=None):
        if params is None:
            params = self.parameters()

        # vector field expects time as input; wrap for odeint
        def vf(y, t, vf_params):
            t_input = jnp.full((y.shape[0], 1), t) if y.ndim > 1 else jnp.array([t])
            x = jnp.concatenate([y, t_input], axis=-1)
            return self.vector_field(x, vf_params)

        y1 = odeint(vf, y0, self.ts, self.vector_field.parameters(), **self.solver_params)
        return y1[-1]  # Return value at final time step

    def parameters(self):
        return self.vector_field.parameters()

    def set_parameters(self, params):
        self.vector_field.set_parameters(params)
