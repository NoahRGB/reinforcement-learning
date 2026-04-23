from .learn import learn, learn_vectorised
from .utils import detect_torch_device, create_tensorboard_writer

__all__ = [
        "learn",
        "learn_vectorised",
        "detect_torch_device",
        "create_tensorboard_writer",
]
