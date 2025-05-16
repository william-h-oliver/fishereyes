from omegaconf import OmegaConf
from fishereyes.models.registry import MODEL_REGISTRY
from fishereyes.losses.registry import LOSS_REGISTRY

class FisherEyes:
    def __init__(self, y0, sigma0):
        self.y0 = y0
        self.sigma0 = sigma0

    @classmethod
    def from_config_files(cls, model_cfg_path, loss_cfg_path):
        model_cfg = OmegaConf.load(model_cfg_path)
        loss_cfg = OmegaConf.load(loss_cfg_path)

        model_cls = MODEL_REGISTRY[model_cfg.name]
        model = model_cls(**model_cfg.params)

        loss_fn = LOSS_REGISTRY[loss_cfg.name](**loss_cfg.params)

        return cls(model=model, loss_fn=loss_fn)