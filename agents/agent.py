from abc import ABC, abstractmethod
import torch

import agents
import envs
import utils

class Agent(ABC):

    @abstractmethod
    def learn(self, total_timesteps: int, env: envs.Gymenv, logger: utils.Logger, seed: int = None, quiet: bool = False):
        ...

    @abstractmethod
    def to(self, device: torch.device):
        # convert all agent torch operations to the given device
        ...