import gymnasium as gym
import numpy as np

class SpecifyActions(gym.ActionWrapper):
    # for some discrete envs, there are loads of useless actions that
    # make agents perform slowly/poorly. this wrapper allows certain
    # actions in a Discrete space to be specified so that useless
    # ones can be ignored

    # e.g. for minigrid 'Empty', the only required actions are left, right,
    # and forward. pickup, drop, toggle and done are all useless and
    # waste time, therefore allowed_actions=[0, 1, 2]
    # https://minigrid.farama.org/environments/minigrid/EmptyEnv/
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = gym.spaces.Discrete(len(allowed_actions))

    def action(self, action):
        return self.allowed_actions[action]
    
class SwapChannel(gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))