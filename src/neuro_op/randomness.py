# Initialize randomness and RNGs
import random

import numpy as np


# Draw 32-bit random seed from entropy source and initialize...
RANDOM_SEED = np.random.SeedSequence().entropy
# ... Python's random module seed (for node selection via random.choice)
rng0 = random.Random(RANDOM_SEED)
# ... Numpy's RNG (primarily for rng.choice, rng.uniform)
rng = np.random.default_rng(RANDOM_SEED)
# ... Numpy's base seed (used by scipy.stats, must be smaller than 2**32)
np.random.seed(RANDOM_SEED % 2**32)


def init_seeds(seed=None):
    """ "(Re-)Initialize randomness and RNGs with provided or new seed."""

    if seed is None:
        seed = np.random.SeedSequence().entropy
    elif not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")
    else:
        pass
    RANDOM_SEED = seed
    rng0 = random.Random(seed)
    rng = np.random.default_rng(seed)
    np.random.seed(seed % 2**32)
    return RANDOM_SEED, rng0, rng


RANDOM_SEED, rng0, rng = init_seeds()
