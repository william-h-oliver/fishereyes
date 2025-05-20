"""
FisherEyes: A flexible framework for learning diffeomorphic transformations that 
normalize heteroskedastic uncertainty.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Standard library imports
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Third-party library imports
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
from tqdm import trange

# Local library imports
from fishereyes.utils.train_utils import loss_and_grad, update
from fishereyes.models.registry import MODEL_REGISTRY
from fishereyes.losses.registry import LOSS_REGISTRY
from fishereyes.optimizers.registry import OPTIMIZER_REGISTRY

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default_config.yaml"


class FisherEyes:
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        opt_state: Any,
        loss_fn: Any,
        epochs: int,
        batch_size: int,
        config: Dict[str, Any] = None,
    ) -> None:
        # Core components
        self.model = model
        self.optimizer = optimizer
        self.opt_state = opt_state
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

        # Save full config for reproducibility/logging
        self.config = config or {}

    @classmethod
    def from_config(
        cls,
        data_dim: int,
        config_path: Optional[Union[str, Path]] = None,
        key: Optional[Union[jax.random.PRNGKey, int]] = None,
    ) -> "FisherEyes":
        """
        Create a FisherEyes instance from a configuration file.

        Parameters:
        - data_dim: Dimensionality of the input/output data.
        - config_path: Path to the configuration file. If None, the default configuration is used.
        - key: Optional jax.random.PRNGKey or integer seed for reproducibility.

        Returns:
        - An instance of the FisherEyes class.
        """
        # === Load the full configuration ===
        config_path = config_path or DEFAULT_CONFIG_PATH
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)

        # === Validate the configuration ===
        if not isinstance(data_dim, int):
            raise TypeError(f"Expected data_dim to be an integer, got {type(data_dim)}.")
        if data_dim <= 0:
            raise ValueError(f"Expected data_dim to be a positive integer, got {data_dim}.")
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

        # === Update model config with input/output dimensions ===
        model_params = dict(config["model"]["params"])  # Make mutable copy
        model_params["input_dim"] = data_dim
        model_params["output_dim"] = data_dim
        if isinstance(key, jax.Array):
            model_params['key'] = key
        elif isinstance(key, int):
            model_params['key'] = jax.random.PRNGKey(key)
        else:
            key = jax.random.PRNGKey(0)

        # === Instantiate model ===
        model_cls = MODEL_REGISTRY[config["model"]["name"]]
        model = model_cls.from_config(model_params)

        # === Instantiate optimizer ===
        optimizer_cls = OPTIMIZER_REGISTRY[config["optimizer"]["name"]]
        optimizer = optimizer_cls(**config["optimizer"]["params"])
        opt_state = optimizer.init(model.parameters())

        # === Instantiate loss function ===
        loss_fn_cls = LOSS_REGISTRY[config["loss"]["name"]]
        loss_fn = loss_fn_cls(**config["loss"]["params"])

        # === Unpack training config ===
        training_cfg = config["training"]
        epochs = training_cfg["epochs"]
        batch_size = training_cfg["batch_size"]

        # === Return initialized instance ===
        return cls(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=batch_size,
            config=config,
        )

    def as_config(self) -> Dict[str, Any]:
        """Return a dictionary representation of the current configuration."""
        return {
            "model": self.model.as_config(),
            "optimizer": {
                "name": self.optimizer.name,
                "params": self.optimizer.params,
            },
            "loss": self.loss_fn.as_config(),
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            },
        }

    def fit(
        self,
        y0: jax.Array,         # shape [N, D]
        sigma0: jax.Array,     # shape [N, D, D]
        key: Optional[Union[jax.random.PRNGKey, int]] = None,
    ) -> None:
        """
        Fit the transformation model to data.

        Parameters:
        - y0: Input data array of shape [N, D]
        - sigma0: Covariance matrices of shape [N, D, D]
        - key: Optional jax.random.PRNGKey or integer seed for reproducibility.
        """
        # === Validate inputs ===
        if not isinstance(y0, jax.Array):
            raise TypeError(f"Expected y0 to be a jax.Array, got {type(y0)}.")
        if not isinstance(sigma0, jax.Array):
            raise TypeError(f"Expected sigma0 to be a jax.Array, got {type(sigma0)}.")
        if y0.ndim != 2:
            raise ValueError(f"Expected y0 to have 2 dimensions, got {y0.ndim}.")
        if sigma0.ndim != 3:
            raise ValueError(f"Expected sigma0 to have 3 dimensions, got {sigma0.ndim}.")
        if y0.shape[0] != sigma0.shape[0]:
            raise ValueError(f"Expected y0 and sigma0 to have the same first dimension, got {y0.shape[0]} and {sigma0.shape[0]}.")
        if y0.shape[1] != sigma0.shape[1] != sigma0.shape[2] != y0.shape[1]:
            raise ValueError(f"Expected y0 and sigma0 to have the same last dimensions, got {y0.shape[1]} and {sigma0.shape[1]}.")
        
        if isinstance(key, jax.Array):
            pass
        elif isinstance(key, int):
            key = jax.random.PRNGKey(key)
        else:
            key = jax.random.PRNGKey(0)

        # Get number of samples
        num_samples = y0.shape[0]

        # NOTE: Last few samples may be dropped if num_samples % batch_size != 0
        steps_per_epoch = num_samples // self.batch_size

        # Retrieve initial state
        params = self.model.parameters()
        opt_state = self.opt_state

        # === Pre-process sigma0 ===
        eigvals0, eigvecs0 = jnp.linalg.eigh(sigma0)

        # === Training loop ===
        term_width = shutil.get_terminal_size((80, 20)).columns
        pbar = trange(self.epochs, desc="Training", ncols=min(term_width, 160))
        for epoch in pbar:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, num_samples)
            y0_shuffled = y0[perm]
            eigvals0_shuffled = eigvals0[perm]
            eigvecs0_shuffled = eigvecs0[perm]

            epoch_loss = 0.0
            for i in range(steps_per_epoch):
                start = i * self.batch_size
                end = start + self.batch_size
                y0_batch = y0_shuffled[start:end]
                eigvals0_batch = eigvals0_shuffled[start:end]
                eigvecs0_batch = eigvecs0_shuffled[start:end]

                loss_val, grads = loss_and_grad(self.model, self.loss_fn, params, y0_batch, eigvals0_batch, eigvecs0_batch)
                params, opt_state = update(self.optimizer, params, opt_state, grads)
                epoch_loss += loss_val

            epoch_loss /= (num_samples // self.batch_size)
            self.loss_history.append(epoch_loss)
            pbar.set_description(f"Epoch {epoch+1:03d}")
            pbar.set_postfix_str(f"loss = {epoch_loss:.6f}")#, relative loss = {100*loss/norm:.1f}%")

        # Update model and optimizer state
        self.model.set_parameters(params)
        self.opt_state = opt_state