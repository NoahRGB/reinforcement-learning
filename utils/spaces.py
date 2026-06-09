
import gymnasium as gym

def detect_space_size(space: gym.Space):
    if type(space) == gym.spaces.Discrete:
        return space.n
    if type(space) == gym.spaces.Box:
        return space.shape
    
def is_space_continuous(space: gym.Space):
    return False if type(space) == gym.spaces.Discrete else True

def is_space_discrete(space: gym.Space):
    return True if type(space) == gym.spaces.Discrete else False