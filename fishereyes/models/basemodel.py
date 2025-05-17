from abc import ABC, abstractmethod
from omegaconf import DictConfig


class ConfigurableModel(ABC):
    """
    Abstract base class for models that can be instantiated from config.
    """

    @classmethod
    def from_config(cls, config: DictConfig, **extra_kwargs):
        """
        Instantiate the model from a nested OmegaConf config dictionary.
        Supports recursive instantiation of submodels if specified.
        """
        from fishereyes.models.registry import MODEL_REGISTRY
        config = dict(config)  # Ensure normal dict

        submodels = {}
        for key, value in config.items():
            if isinstance(value, dict) and "name" in value and "params" in value:
                submodel_cls = MODEL_REGISTRY[value["name"]]
                submodels[key] = submodel_cls.from_config(value["params"], **extra_kwargs)
            else:
                submodels[key] = value

        return cls(**submodels, **extra_kwargs)
    
    @abstractmethod
    def init_parameters(self, input_dim: int, key):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, params):
        pass
