# Standard imports
from typing import Optional, Union, Dict, Any

# Third-party imports
import jax


def validate_data_dim(data_dim: int) -> None:
    """
    Validate the dimensionality of the input/output data.
    Parameters:
    - data_dim: Dimensionality of the input/output data.
    """
    if not isinstance(data_dim, int):
        raise TypeError(f"Expected data_dim to be an integer, got {type(data_dim)}.")
    if data_dim <= 0:
        raise ValueError(f"Expected data_dim to be a positive integer, got {data_dim}.")
    
def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary.
    Parameters:
    - config: Configuration dictionary.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Expected config to be a dictionary, got {type(config)}.")
    for key in ["model", "optimizer", "loss", "training"]:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in configuration.")
        if not isinstance(config[key], dict):
            raise TypeError(f"Expected '{key}' to be a dictionary, got {type(config[key])}.")
        if key == "training":
            for subkey in ["epochs", "batch_size"]:
                if subkey not in config[key]:
                    raise KeyError(f"Missing required key '{subkey}' in training configuration.")
                if not isinstance(config[key][subkey], int):
                    raise TypeError(f"Expected '{subkey}' to be an integer, got {type(config[key][subkey])}.")
                if config[key][subkey] <= 0:
                    raise ValueError(f"Expected '{subkey}' to be a positive integer, got {config[key][subkey]}.")
        else:
            for subkey in ["name", "params"]:
                if subkey not in config[key]:
                    raise KeyError(f"Missing required key '{subkey}' in '{key}' configuration.")
            if not isinstance(config[key]["name"], str):
                raise TypeError(f"Expected 'name' to be a string, got {type(config[key]['name'])}.")
            if not isinstance(config[key]["params"], dict):
                raise TypeError(f"Expected 'params' to be a dictionary, got {type(config[key]['params'])}.")
            
def validate_key(key: Optional[Union[jax.random.key, int]]) -> None:
    """
    Validate the key for reproducibility.
    Parameters:
    - key: Optional jax.random.key or integer seed for reproducibility.
    """
    if key is not None and not isinstance(key, (jax.Array, int)):
        raise TypeError(f"Expected key to be a jax.random.key or an integer, got {type(key)}.")
        
def validate_input_data(
    y0: jax.Array,
    sigma0: jax.Array
) -> None:
    if not isinstance(y0, jax.Array):
        raise TypeError(f"Expected y0 to be a jax.Array, got {type(y0)}.")
    if y0.ndim != 2:
        raise ValueError(f"Expected y0 to have 2 dimensions, got {y0.ndim}.")
    if not isinstance(sigma0, jax.Array):
        raise TypeError(f"Expected sigma0 to be a jax.Array, got {type(sigma0)}.")
    if sigma0.ndim != 3:
        raise ValueError(f"Expected sigma0 to have 3 dimensions, got {sigma0.ndim}.")
    if y0.shape[0] != sigma0.shape[0]:
        raise ValueError(f"Expected y0 and sigma0 to have the same first dimension, got {y0.shape[0]} and {sigma0.shape[0]}.")
    if y0.shape[1] != sigma0.shape[1] or sigma0.shape[1] != sigma0.shape[2]:
        raise ValueError(f"Expected y0 and sigma0 to have compatible dimensions, got {y0.shape[1]} and {sigma0.shape[1]}.")
