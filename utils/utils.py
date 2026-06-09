import os
import random
import torch
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

tensorboard = True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    tensorboard = False

def detect_torch_device(quiet=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not quiet: print(f"using device {device}")
    return device

def create_tensorboard_writer(comment="", flush_secs=5):
    if tensorboard:
        return SummaryWriter(comment=comment, flush_secs=flush_secs)
    return None

def seed(seed: int):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)