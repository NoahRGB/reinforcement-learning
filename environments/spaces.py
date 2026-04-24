import gymnasium as gym

from enum import Enum

class EnvType(Enum):
    SINGULAR = 1
    VECTORISED = 2
    
class DiscreteSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.num = 1

class ContinuousSpace:
    def __init__(self, dimensions, min_bounds, max_bounds, dtype):
        self.dimensions = dimensions
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.dtype = dtype
        self.num = len(dimensions)

def detect_space(space):

    if type(space) is gym.spaces.Discrete:
        return DiscreteSpace(space.n)
    
    if type(space) is gym.spaces.Box:
        return ContinuousSpace(space.shape, space.low, space.high, space.dtype)
