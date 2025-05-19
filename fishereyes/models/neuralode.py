import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from typing import Optional, Union, Any
from omegaconf import DictConfig

from fishereyes.models.basemodel import ConfigurableModel


class NeuralODE(ConfigurableModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        vector_field: ConfigurableModel,
        time_dependence: bool = True,
        time_length: float = 1.0,
        time_steps: int = 10,
        solver_params: Optional[dict] = None,
        key: Optional[Union[jax.random.PRNGKey, int]] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vector_field = vector_field
        self.time_dependence = time_dependence
        self.time_length = time_length
        self.time_steps = time_steps
        self.solver_params = solver_params or {}
        if isinstance(key, jax.Array):
            self.key = key
        elif isinstance(key, int):
            self.key = jax.random.PRNGKey(key)
        else:
            self.key = jax.random.PRNGKey(0)

        self.ts = jnp.linspace(0.0, time_length, time_steps)
    
    @classmethod
    def from_config(cls, config: DictConfig, **extra_kwargs):
        """
        Custom from_config for NeuralODE to handle dynamic input_dim logic and submodel instantiation.
        """
        from fishereyes.models.registry import MODEL_REGISTRY

        config = dict(config)  # Convert to standard dict
        constructor_dict = {}

        for key, value in config.items():
            if key == "vector_field" and isinstance(value, dict) and "name" in value and "params" in value:
                # Adjust vector field input_dim based on time_dependence flag
                time_dependence = config.get("time_dependence", True)
                constructor_dict["time_dependence"] = time_dependence

                input_dim = config["input_dim"] + (1 if time_dependence else 0)
                output_dim = config["output_dim"]

                vector_field_cls = MODEL_REGISTRY[value["name"]]
                vector_field_config = dict(value["params"])
                vector_field_config.update(dict(input_dim=input_dim, output_dim=output_dim))

                constructor_dict[key] = vector_field_cls.from_config(vector_field_config, **extra_kwargs)
            else:
                constructor_dict[key] = value

        return cls(**constructor_dict, **extra_kwargs)


    def __call__(
        self,
        y0: jax.Array,
        ts: Optional[jax.Array] = None,
        params: Optional[Any] = None,
    ) -> jax.Array:
        ts = self.ts if ts is None else ts
        params = self.parameters() if params is None else params
        return odeint(self._wrapped_vector_field, y0, ts, params)

    def _wrapped_vector_field(
        self,
        y: jax.Array,
        t: float,
        params: Optional[Any]
    ) -> jax.Array:
        if self.time_dependence:
            # Add time as an additional input to the vector field
            time_input = jnp.full((y.shape[0], 1), t)
            input_vector = jnp.concatenate((y, time_input), axis=1)
        else:
            input_vector = y
        return self.vector_field(input_vector, params=params["vector_field"])
    
    def parameters(self) -> Any:
        return {
            "vector_field": self.vector_field.parameters()
        }

    def set_parameters(self, params: Any) -> None:
        self.vector_field.set_parameters(params["vector_field"])

