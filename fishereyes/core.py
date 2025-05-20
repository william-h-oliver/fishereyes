"""
FisherEyes: A flexible framework for learning diffeomorphic transformations that 
normalize heteroskedastic uncertainty.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Standard library imports
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List, Any

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
        key: Optional[Union[jax.random.key, int]] = None,
    ) -> "FisherEyes":
        """
        Create a FisherEyes instance from a configuration file.

        Parameters:
        - data_dim: Dimensionality of the input/output data.
        - config_path: Path to the configuration file. If None, the default configuration is used.
        - key: Optional jax.random.key or integer seed for reproducibility.

        Returns:
        - An instance of the FisherEyes class.
        """
        # === Load the full configuration ===
        config_path = config_path or DEFAULT_CONFIG_PATH
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)

        # === Validate the configuration ===
        cls._validate_from_config_inputs(data_dim, config, key)

        # === Update model config with input/output dimensions ===
        model_params = dict(config["model"]["params"])  # Make mutable copy
        model_params["input_dim"] = data_dim
        model_params["output_dim"] = data_dim
        if isinstance(key, jax.Array):
            model_params['key'] = key
        elif isinstance(key, int):
            model_params['key'] = jax.random.key(key)
        else:
            key = jax.random.key(0)

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
    
    @classmethod
    def _validate_from_config_inputs(
        cls,
        data_dim: int,
        config: Dict[str, Any],
        key: Optional[Union[jax.random.key, int]] = None,
    ) -> None:
        """
        Validate the inputs for the from_config method.
        Parameters:
        - data_dim: Dimensionality of the input/output data.
        - config: Configuration dictionary.
        - key: Optional jax.random.key or integer seed for reproducibility.
        """
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
        key: Optional[Union[jax.random.key, int]] = None,
    ) -> None:
        """
        Fit the transformation model to data.

        Parameters:
        - y0: Input data array of shape [N, D]
        - sigma0: Covariance matrices of shape [N, D, D]
        - key: Optional jax.random.key or integer seed for reproducibility.
        """
        # === Validate inputs ===
        self._validate_fit_inputs(y0, sigma0, key)
        
        if isinstance(key, int):
            key = jax.random.key(key)
        elif key is None:
            key = jax.random.key(0)

        params = self.model.parameters()
        opt_state = self.opt_state
        eigvals0, eigvecs0 = jnp.linalg.eigh(sigma0)

        terminal_width = shutil.get_terminal_size((80, 20)).columns
        pbar = trange(self.epochs, desc="Training", ncols=min(terminal_width, 160))
        for epoch in pbar:
            epoch_loss, params, opt_state, key = self._train_epoch(y0, eigvals0, eigvecs0, params, opt_state, key)
            self.loss_history.append(epoch_loss)
            self._update_progress_bar(pbar, epoch, epoch_loss)

        self.model.set_parameters(params)
        self.opt_state = opt_state

    def _validate_fit_inputs(
        self,
        y0: jax.Array,
        sigma0: jax.Array,
        key: Optional[Union[jax.random.key, int]] = None,
    ) -> None:
        """
        Validate the inputs for the fit method.
        Parameters:
        - y0: Input data array of shape [N, D]
        - sigma0: Covariance matrices of shape [N, D, D]
        - key: Optional jax.random.key or integer seed for reproducibility.
        """
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
        if key is not None and not isinstance(key, (jax.Array, int)):
            raise TypeError(f"Expected key to be a jax.Array or int, got {type(key)}.")
        
    def _shuffle_data(
        self,
        y0: jax.Array,
        eigvals0: jax.Array,
        eigvecs0: jax.Array,
        key: jax.random.key,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.random.key]:
        """
        Shuffle the data and return the shuffled arrays along with the new key.
        Parameters:
        - y0: Input data array of shape [N, D]
        - eigvals0: Eigenvalues array of shape [N, D]
        - eigvecs0: Eigenvectors array of shape [N, D, D]
        - key: jax.random.key for random operations.
        Returns:
        - Shuffled y0, eigvals0, eigvecs0, and the new key.
        """
        # Shuffle the data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, y0.shape[0])
        return y0[perm], eigvals0[perm], eigvecs0[perm], key
    
    def _process_batch(
        self,
        params: jax.Array,
        opt_state: jax.Array,
        y0_batch: jax.Array,
        eigvals0_batch: jax.Array,
        eigvecs0_batch: jax.Array
    ) -> Tuple[float, jax.Array, jax.Array]:
        """
        Process a batch of data and update the model parameters.
        Parameters:
        - params: Model parameters.
        - opt_state: Optimizer state.
        - y0_batch: Input data batch of shape [batch_size, D]
        - eigvals0_batch: Eigenvalues batch of shape [batch_size, D]
        - eigvecs0_batch: Eigenvectors batch of shape [batch_size, D, D]
        Returns:
        - loss_val: Loss value for the batch.
        - params: Updated model parameters.
        - opt_state: Updated optimizer state.
        """
        # Compute loss and gradients
        loss_val, grads = loss_and_grad(self.model, self.loss_fn, params, y0_batch, eigvals0_batch, eigvecs0_batch)
        
        # Update parameters and optimizer state
        params, opt_state = update(self.optimizer, params, opt_state, grads)
        
        return loss_val, params, opt_state
    
    def _train_epoch(
        self,
        y0: jax.Array,
        eigvals0: jax.Array,
        eigvecs0: jax.Array,
        params: jax.Array,
        opt_state: jax.Array,
        key: jax.random.key,
    ) -> Tuple[float, jax.Array, jax.Array, jax.random.key]:
        """
        Train the model for one epoch.
        """
        y0_batches, eigvals0_batches, eigvecs0_batches, key = self._shuffle_and_split_batches(y0, eigvals0, eigvecs0, key)
        epoch_loss = 0.0
        for y0_batch, eigvals0_batch, eigvecs0_batch in zip(y0_batches, eigvals0_batches, eigvecs0_batches):
            loss_val, params, opt_state = self._process_batch(params, opt_state, y0_batch, eigvals0_batch, eigvecs0_batch)
            epoch_loss += loss_val
        return epoch_loss / len(y0_batches), params, opt_state, key
    
    def _shuffle_and_split_batches(
        self,
        y0: jax.Array,
        eigvals0: jax.Array,
        eigvecs0: jax.Array,
        key: jax.random.key,
    ) -> Tuple[List[jax.Array], List[jax.Array], List[jax.Array], jax.random.key]:
        """
        Shuffle the data and split it into batches.
        """
        y0, eigvals0, eigvecs0, key = self._shuffle_data(y0, eigvals0, eigvecs0, key)
        y0_batches = [y0[i:i + self.batch_size] for i in range(0, y0.shape[0], self.batch_size)]
        eigvals0_batches = [eigvals0[i:i + self.batch_size] for i in range(0, eigvals0.shape[0], self.batch_size)]
        eigvecs0_batches = [eigvecs0[i:i + self.batch_size] for i in range(0, eigvecs0.shape[0], self.batch_size)]
        return y0_batches, eigvals0_batches, eigvecs0_batches, key

    def _update_progress_bar(
        self,
        pbar: Any,
        epoch: int,
        epoch_loss: float,
    ) -> None:
        """
        Update the progress bar with the current epoch and loss.
        Parameters:
        - pbar: The progress bar object.
        - epoch: Current epoch number.
        - epoch_loss: Loss value for the current epoch.
        """
        # Update progress bar
        pbar.set_description(f"Epoch {epoch+1:03d}")
        pbar.set_postfix_str(f"loss = {epoch_loss:.6f}")