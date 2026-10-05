"""Isolate the upstream global RNGs without editing the pinned implementation."""
from contextlib import contextmanager
import random

from .protocol import domain_seed


@contextmanager
def random_domain(seed, domain, event):
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    value = domain_seed(seed, domain, event)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if cuda_states is not None:
        torch.cuda.manual_seed_all(value)
    try:
        yield value
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
