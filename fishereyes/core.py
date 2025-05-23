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
        config: Dict[str, Any] = None,
    ) -> None:
        # Core components
        self.model = model
        self.optimizer = optimizer
        self.opt_state = opt_state
        self.loss_fn = loss_fn
        self.epochs = epochs
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
        if epochs == "None":
            epochs = 100

        # === Return initialized instance ===
        return cls(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            epochs=epochs,
            config=config,
        )

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

        # === Initialize parameters ===
        params = self.model.parameters()
        opt_state = self.opt_state
        self.loss_fn.calculate_optimal_alpha(eigvals0)

        # === Initialize progress bar ===
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        pbar = trange(self.epochs, desc="Training", ncols=min(terminal_width, 160))
        reference_loss = self.loss_fn.calculate_reference_loss(eigvals0)

        # === Training loop ===
        for epoch in pbar:
            # Compute loss and gradients
            epoch_loss, grads = loss_and_grad(self.model, self.loss_fn, params, y0, eigvals0, eigvecs0)
            
            # Update parameters and optimizer state
            params, opt_state = update(self.optimizer, params, opt_state, grads)
            
            # Store loss history
            self.loss_history.append(epoch_loss)

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1:03d}")
            pbar.set_postfix_str(f"loss = {epoch_loss:.6f}, rel_loss = {100 * epoch_loss / reference_loss:.1f}%")

        # === Finalize training ===
        self.model.set_parameters(params)
        self.opt_state = opt_state
    
    def predict(
        self,
        y0: jax.Array,         # shape [N, D]
        sigma0: jax.Array,     # shape [N, D, D]
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> jax.Array:
        """
        Predict the transformed data.

        Parameters:
        - y0: Input data array of shape [N, D]
        - sigma0: Covariance matrices of shape [N, D, D]
        - kwargs: Optional additional arguments for the model

        Returns:
        - Transformed data array of shape [N, D]
        - Transformed covariance matrices of shape [N, D, D]
        """
        # === Validate inputs ===
        validate_input_data(y0, sigma0)
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TypeError(f"Expected kwargs to be a dictionary, got {type(kwargs)}.")

        # === Transform point data ===
        y1 = self.model(y0, **kwargs)

        # === Compute Jacobians ===
        def single_jac(y):
            return jax.jacrev(lambda x: self.model(x, **kwargs))(y)  # shape (D, D)
        J = jax.vmap(single_jac)(y0)  # shape (N, D, D)

        # === Transform covariance matrices ===
        sigma1 = jnp.einsum('nid,ndk,njd->nij', J, sigma0, J) # shape (N, D, D)
        
        return y1, sigma1