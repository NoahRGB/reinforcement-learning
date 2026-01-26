from abc import ABC, abstractmethod


class Environment(ABC):

    @abstractmethod
    def step(self, s, a):
        ...

    @abstractmethod
    def get_start_state(self):
        ...

    @abstractmethod
    def get_state_space(self):
        ...

    @abstractmethod
    def get_action_space(self):
        ...

    @abstractmethod
    def reset(self):
        ...
