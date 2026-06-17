from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):

    @abstractmethod
    def step(self, actions: np.array):
        ...

    @abstractmethod
    def get_num_envs(self):
        ...

    @abstractmethod
    def get_single_state_space(self):
        ...

    @abstractmethod
    def get_single_action_space(self):
        ...

    @abstractmethod
    def get_start_states(self):
        ...

    @abstractmethod
    def is_conv(self):
        ...