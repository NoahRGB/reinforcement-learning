import numpy as np

import torch

tensorboard = True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    tensorboard = False

def detect_torch_device():
    # return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tensorboard_writer(comment="", flush_secs=5):
    if tensorboard:
        return SummaryWriter(comment=comment, flush_secs=flush_secs)
    return None

def smoothing(vals, factor):
    # https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
    last_smoothed_val = vals[0]
    smoothed_vals = np.zeros(len(vals))
    for i, val in enumerate(vals):
        smoothed_val = last_smoothed_val * factor + (1 - factor) * val
        smoothed_vals[i] = smoothed_val
        last_smoothed_val = smoothed_val
    return smoothed_vals
