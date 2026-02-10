import torch

from tensorboardX import SummaryWriter

def detect_torch_device():
    # return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tensorboard_writer(comment="", flush_secs=5):
    return SummaryWriter(comment=comment, flush_secs=flush_secs)
