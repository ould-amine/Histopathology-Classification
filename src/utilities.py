import random
import numpy as np
import torch


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dictionary.

    Example:
        {"a": {"b": 1}} → {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    """
    # === Python / NumPy / Torch ===
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # === Deterministic behavior ===
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
