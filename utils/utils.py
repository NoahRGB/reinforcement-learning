import torch

from tensorboardX import SummaryWriter

def detect_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tensorboard_writer(comment=""):
    return SummaryWriter(comment=comment)
