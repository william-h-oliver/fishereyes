# Third-party imports
import optax

def Adam(**kwargs) -> optax.GradientTransformation:
    if 'lr' in kwargs:
        kwargs['learning_rate'] = kwargs.pop('lr')
    return optax.adam(**kwargs)