import gc
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_color(content: str, color: str = "green") -> None:
    """Print with an ANSI color hint. Falls back to plain print if no terminal."""
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "blue": "\033[94m"}
    reset = "\033[0m"
    prefix = colors.get(color, "")
    print(f"{prefix}{content}{reset}")
