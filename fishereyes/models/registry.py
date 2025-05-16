from fishereyes.models.mlp import MLP
from fishereyes.models.neural_ode import NeuralODE

MODEL_REGISTRY = {
    "mlp": MLP,
    "neural_ode": NeuralODE,
}