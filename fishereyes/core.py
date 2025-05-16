from omegaconf import OmegaConf

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
        config,
    ):
        # Core components
        self.model = model
        self.optimizer = optimizer
        self.opt_state = opt_state
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size

        # Save full config for reproducibility/logging
        self.config = config

    @classmethod
    def from_config(cls, config_path):
        # Load the full configuration
        config = OmegaConf.load(config_path)

        # === Instantiate model ===
        model_cls = MODEL_REGISTRY[config.model.name]
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
