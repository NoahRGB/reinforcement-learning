from environment import Environment

import gymnasium as gym

class GymEnvironment(Environment):
    def __init__(self, name, rendermode):
        self.env = gym.make(name, render_mode=rendermode) 

    def __del__(self):
        self.env.close() 

    def step(self, action):
        return self.env.step(action)

    def get_state_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space

