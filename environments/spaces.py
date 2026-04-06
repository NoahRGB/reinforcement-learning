import gymnasium as gym

from enum import Enum

class EnvType(Enum):
    SINGULAR = 1
    VECTORISED = 2
    
class DiscreteSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions

class ContinuousSpace:
    def __init__(self, dimensions, min_bounds, max_bounds, dtype):
        self.dimensions = dimensions
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.dtype = dtype

def detect_space(space):

    if type(space) is gym.spaces.Discrete:
        return DiscreteSpace(space.n)
    
    if type(space) is gym.spaces.Box:
        return ContinuousSpace(space.shape, space.low, space.high, space.dtype)
