# Standard imports
from typing import Optional, Union, Dict, Any

# Third-party imports
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

# Local imports
from fishereyes.models.basemodel import ConfigurableModel
from fishereyes.utils.key_utils import create_key


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
        key: Optional[Union[jax.random.key, int]] = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vector_field = vector_field
        self.time_dependence = time_dependence
        self.time_length = time_length
        self.time_steps = time_steps
        self.solver_params = solver_params or {}
        self.key = create_key(key)

        self.ts = jnp.linspace(0.0, time_length, time_steps)
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        **extra_kwargs
    ) -> "NeuralODE":
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
                vector_field_key = config.get("key", 0)
                if isinstance(vector_field_key, int):
                    vector_field_key = jax.random.key(vector_field_key)
                vector_field_key, _ = jax.random.split(vector_field_key)

                vector_field_cls = MODEL_REGISTRY[value["name"]]
                vector_field_config = dict(value["params"])
                vector_field_config.update(dict(input_dim=input_dim, output_dim=output_dim, key=vector_field_key))

                constructor_dict[key] = vector_field_cls.from_config(vector_field_config, **extra_kwargs)
            else:
                constructor_dict[key] = value

        return cls(**constructor_dict, **extra_kwargs)

    def __call__(
        self,
        y0: jax.Array,
        ts: Optional[jax.Array] = None,
        params: Optional[Dict[str, Any]] = None,
        final_state_only: bool = True
    ) -> jax.Array:
        ts = self.ts if ts is None else ts
        params = self.parameters() if params is None else params
        paths = odeint(self._wrapped_vector_field, y0, ts, params)
        if final_state_only:
            return paths[-1]
        else:
            return paths

    def _wrapped_vector_field(
        self,
        y: jax.Array,
        t: float,
        params: Dict[str, Any],
    ) -> jax.Array:
        if self.time_dependence:
            # Add time as an additional input to the vector field
            time_input = jnp.full((y.shape[0], 1), t) if y.ndim > 1 else jnp.array([t])
            input_vector = jnp.concatenate((y, time_input), axis=-1)
        else:
            input_vector = y
        return self.vector_field(input_vector, params=params["vector_field"])
    
    def parameters(self) -> Dict[str, Any]:
        return {
            "vector_field": self.vector_field.parameters()
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.vector_field.set_parameters(params["vector_field"])

