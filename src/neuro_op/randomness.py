# Initialize randomness and RNGs
import random

import numpy as np


RANDOM_SEED = np.random.SeedSequence().entropy
rng0 = random.Random(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
