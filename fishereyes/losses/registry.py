from fishereyes.losses.ssi_kl import SymmetrizedScaleInvariantKL

LOSS_REGISTRY = {
    "ssiKLdiv": SymmetrizedScaleInvariantKL,
}