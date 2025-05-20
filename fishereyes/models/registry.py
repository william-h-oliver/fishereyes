# Local imports
from fishereyes.models.mlp import MLP
from fishereyes.models.neuralode import NeuralODE

MODEL_REGISTRY = {
    "mlp": MLP,
    "neural_ode": NeuralODE,
}