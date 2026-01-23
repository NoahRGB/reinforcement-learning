import random
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from environment import Environment

class Agent(ABC):
    def __init__(self, environment: Environment):
        self.environment = environment

    @abstractmethod
    def run_policy(self, state):
        pass

    @abstractmethod
    def learn(self, iterations, quiet=False):
        pass

    @abstractmethod
    def __str__(self):
        return str("")
