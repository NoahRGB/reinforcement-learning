from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def run_policy(self, s):
        ...

    @abstractmethod
    def update(self, s, sprime, a, r):
        ...

    @abstractmethod
    def initialise(self):
        ...

    @abstractmethod
    def finish_episode(self):
        ...
