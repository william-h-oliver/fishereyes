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
import optax
from tqdm import trange

# Local library imports
from fishereyes.models.registry import MODEL_REGISTRY
from fishereyes.losses.registry import LOSS_REGISTRY
from fishereyes.utils.train_utils import (
    shuffle_and_split_batches,
    loss_and_grad,
    update,
)
from fishereyes.utils.validation_utils import (
    validate_data_dim,
    validate_config,
    validate_key,
    validate_input_data,
)
from fishereyes.utils.key_utils import create_key

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default_config.yaml"


class FisherEyes:
    """
    FisherEyes: A class for learning diffeomorphic transformations that normalize
    heteroskedastic uncertainty.

    Parameters
    ----------
    model: Any
        The model to be trained.
    optimizer: Any
        The optimizer to be used for training.
    opt_state: Any
        The initial state of the optimizer.
    loss_fn: Any
        The loss function to be used for training.
    epochs: int
        The number of epochs to train the model.
    batch_size: int
        The size of the batches to be used during training.
    config: Dict[str, Any], optional
        A dictionary containing the configuration for the model, optimizer, and loss function.
        If None, the default configuration is used.

    Attributes
    ----------
    loss_history: List[float]
        A list containing the loss values for each epoch during training.
    """
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
        self.config = config or {}
        
        self.loss_history = []

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
        validate_data_dim(data_dim)
        validate_config(config)
        validate_key(key)

        # === Update model config with input/output dimensions ===
        model_params = dict(config["model"]["params"])  # Make mutable copy
        model_params["input_dim"] = data_dim
        model_params["output_dim"] = data_dim
        model_params['key'] = create_key(key)

        # === Instantiate model ===
        model_cls = MODEL_REGISTRY[config["model"]["name"]]
        model = model_cls.from_config(model_params)

        # === Instantiate optimizer ===
        optimizer_cls = getattr(optax, config["optimizer"]["name"])
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
        validate_input_data(y0, sigma0)
        validate_key(key)

        # === Preprocess data ===
        eigvals0, eigvecs0 = jnp.linalg.eigh(sigma0)
        
        # === Initialize key ===
        key = create_key(key)

        # === Initialize model parameters ===
        params = self.model.parameters()
        opt_state = self.opt_state

        # === Initialize progress bar ===
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        pbar = trange(self.epochs, desc="Training", ncols=min(terminal_width, 160))

        # === Training loop ===
        for epoch in pbar:
            # Shuffle and split data into batches
            y0_batches, eigvals0_batches, eigvecs0_batches, key = shuffle_and_split_batches(self.batch_size, y0, eigvals0, eigvecs0, key)
            
            # Loop over batches
            epoch_loss = 0.0
            for y0_batch, eigvals0_batch, eigvecs0_batch in zip(y0_batches, eigvals0_batches, eigvecs0_batches):
                # Compute loss and gradients
                loss_val, grads = loss_and_grad(self.model, self.loss_fn, params, y0_batch, eigvals0_batch, eigvecs0_batch)
                epoch_loss += loss_val * len(y0_batch)  # Accumulate loss over batches
                
                # Update parameters and optimizer state
                params, opt_state = update(self.optimizer, params, opt_state, grads)
            
            # Average loss over the epoch
            epoch_loss /= y0.shape[0]  # Normalize by number of samples
            self.loss_history.append(epoch_loss)

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1:03d}")
            pbar.set_postfix_str(f"loss = {epoch_loss:.6f}")

        # === Finalize training ===
        self.model.set_parameters(params)
        self.opt_state = opt_state