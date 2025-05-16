import optax

def Adam(**kwargs):
    if 'lr' in kwargs:
        kwargs['learning_rate'] = kwargs.pop('lr')
    return optax.adam(**kwargs)