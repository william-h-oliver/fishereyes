# FisherEyes Core Module

# Standard library imports
import shutil

# Third-party library imports
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
from typing import Optional
from jax.random import PRNGKey
from fishereyes.utils.train_utils import loss_and_grad, update
from tqdm import trange

# Local library imports
from fishereyes.models.registry import MODEL_REGISTRY
from fishereyes.losses.registry import LOSS_REGISTRY
from fishereyes.optimizers.registry import OPTIMIZER_REGISTRY


class FisherEyes:
    def __init__(
        self,
        model,
        optimizer,
        opt_state,
        loss_fn,
        epochs,
        batch_size,
        config={},
    ):
        # Core components
        self.model = model
        self.optimizer = optimizer
        self.opt_state = opt_state
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

        # Save full config for reproducibility/logging
        self.config = config

    @classmethod
    def from_config(cls, config_path):
        # Load the full configuration
        config = OmegaConf.load(config_path)

        # === Instantiate model ===
        model_cls = MODEL_REGISTRY[config.model.name]
        if hasattr(model_cls, "from_config"):
            data_dim = config.training.get("data_dim", 2)
            model = model_cls.from_config(dict(config.model.params), data_dim=data_dim)
        else:
            model = model_cls(**config.model.params)

        # === Instantiate optimizer ===
        optimizer_cls = OPTIMIZER_REGISTRY[config.optimizer.name]
        optimizer = optimizer_cls(**config.optimizer.params)
        opt_state = optimizer.init(model.parameters())

        # === Instantiate loss function ===
        loss_fn = LOSS_REGISTRY[config.loss.name](**config.loss.params)

        # === Unpack training config ===
        training_cfg = config.training
        epochs = training_cfg.epochs
        batch_size = training_cfg.batch_size

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

    def as_config(self):
        """Return a dictionary representation of the current configuration."""
        return {
            "model": {
                "name": self.config.model.name,
                "params": dict(self.config.model.params),
            },
            "optimizer": {
                "name": self.config.optimizer.name,
                "params": dict(self.config.optimizer.params),
            },
            "loss": {
                "name": self.config.loss.name,
                "params": dict(self.config.loss.params),
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            },
        }

    def fit(
        self,
        y0: jnp.ndarray,         # shape [N, D]
        sigma0: jnp.ndarray,     # shape [N, D, D]
        key: Optional[PRNGKey] = None,
    ) -> None:
        """
        Fit the transformation model to data.

        Parameters:
        - y0: Input data array of shape [N, D]
        - sigma0: Covariance matrices of shape [N, D, D]
        - key: Optional PRNGKey for reproducibility
        """
        if key is None: key = jax.random.PRNGKey(42)

        # Get number of samples
        num_samples = y0.shape[0]

        # NOTE: Last few samples may be dropped if num_samples % batch_size != 0
        steps_per_epoch = num_samples // self.batch_size

        # Retrieve initial state
        params = self.model.parameters()
        if params is None:
            key, subkey = jax.random.split(key)
            params = self.model.init_parameters(y0.shape[1], subkey)
            self.model.set_parameters(params)
        opt_state = self.opt_state

        # === Training loop ===
        term_width = shutil.get_terminal_size((80, 20)).columns
        pbar = trange(self.epochs, desc="Training", ncols=min(term_width, 160))
        for epoch in pbar:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, num_samples)
            y0_shuffled = y0[perm]
            sigma0_shuffled = sigma0[perm]

            epoch_loss = 0.0
            for i in range(steps_per_epoch):
                start = i * self.batch_size
                end = start + self.batch_size
                y0_batch = y0_shuffled[start:end]
                sigma0_batch = sigma0_shuffled[start:end]

                loss_val, grads = loss_and_grad(self.model, self.loss_fn, params, y0_batch, sigma0_batch)
                params, opt_state = update(self.optimizer, params, opt_state, grads)
                epoch_loss += loss_val

            epoch_loss /= (num_samples // self.batch_size)
            self.loss_history.append(epoch_loss)
            pbar.set_description(f"Epoch {epoch+1:03d}")
            pbar.set_postfix_str(f"loss = {epoch_loss:.6f}")#, relative loss = {100*loss/norm:.1f}%")

        # Update model and optimizer state
        self.model.set_parameters(params)
        self.opt_state = opt_state