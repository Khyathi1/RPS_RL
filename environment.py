import numpy as np


def opponent_action(strategy, previous, rng):
    """Opponents: random, repeat70, cycle, or biased_rock."""
    if strategy == 'random':
        return int(rng.integers(3))
    if strategy == 'repeat70':
        return previous if rng.random() < 0.7 else int(rng.integers(3))
    if strategy == 'cycle':
        return (previous + 1) % 3
    if strategy == 'biased_rock':
        return int(rng.choice(3, p=[0.7, 0.15, 0.15]))
    raise ValueError(f'Unknown opponent strategy: {strategy}')
