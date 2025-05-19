from abc import ABC, abstractmethod
from omegaconf import DictConfig


class ConfigurableModel(ABC):
    """
    Abstract base class for models that can be instantiated from config.
    """

    @classmethod
    def from_config(cls, config: DictConfig, **extra_kwargs):
        """
        Default implementation of `from_config` for models where:
        - Each submodel is specified by a {"name": ..., "params": ...} dict.
        - There is no conditional logic or parameter adaptation needed.
        Override this method in subclasses if the model requires interdependent parameter logic.
        """
        from fishereyes.models.registry import MODEL_REGISTRY
        config = dict(config)  # Convert to standard dict

        constructor_dict = {}
        for key, value in config.items():
            if isinstance(value, dict) and "name" in value and "params" in value:
                submodel_cls = MODEL_REGISTRY[value["name"]]
                constructor_dict[key] = submodel_cls.from_config(value["params"], **extra_kwargs)
            else:
                constructor_dict[key] = value

        return cls(**constructor_dict, **extra_kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, params):
        pass
