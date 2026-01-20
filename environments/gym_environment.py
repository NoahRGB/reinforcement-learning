from environments.environment import Environment

import gymnasium as gym

class GymEnvironment(Environment):
    def __init__(self, name, **kwargs):
        self.env = gym.make(name, **kwargs)

    def __del__(self):
        self.env.close()

    def step(self, s, a):
        sprime, r, is_terminated, is_truncated, info = self.env.step(a)
        return sprime, r, (is_terminated or is_truncated)

    def reset(self):
        self.env.reset()

    def get_start_state(self):
        start_state, _ = self.env.reset()
        return start_state

    def get_state_space_size(self):
        return self.env.observation_space.n

    def get_action_space_size(self):
        return self.env.action_space.n
