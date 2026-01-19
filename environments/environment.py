from abc import ABCMeta, abstractmethod

class Environment(ABCMeta):
    
    @abstractmethod
    def step(self, state):
        ...

    @abstractmethod
    def get_state_space(self):
        ...

    @abstractmethod
    def get_action_space(self):
        ...
