from .learn import learn, learn_vectorised
from .evaluate import evaluate
from .utils import detect_torch_device, create_tensorboard_writer

__all__ = [
        "learn",
        "learn_vectorised",
        "evaluate",
        "detect_torch_device",
        "create_tensorboard_writer",
]
