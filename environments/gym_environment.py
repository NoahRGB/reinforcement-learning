from environments.environment import Environment
from environments.spaces import detect_space, EnvType

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import numpy as np

class GymEnvironment(Environment):
    def __init__(self, name, is_recording, **kwargs):
        if is_recording and kwargs["render_mode"] == None:
            kwargs["render_mode"] = "rgb_array"

        self.env = gym.make(name, **kwargs)
        if is_recording:
            self.env = RecordVideo(self.env, ".", episode_trigger=lambda x: True)

        self.state_space = detect_space(self.env.observation_space)
        self.action_space = detect_space(self.env.action_space)
        self.num_envs = 1

    def __del__(self):
        self.env.close()

    def step(self, s, a):
        chosen_action = a
        if type(chosen_action) == np.ndarray:
            chosen_action = a[0]
        sprime, r, is_terminated, is_truncated, info = self.env.step(chosen_action)
        return np.expand_dims(sprime, axis=0), np.array([r]), np.array([is_terminated or is_truncated])

    def reset(self):
        self.env.reset()

    def get_start_state(self):
        start_state, _ = self.env.reset()
        return np.expand_dims(start_state, axis=0) # add a fake dimension to represent the 1 environment
    
    def get_env_type(self):
        return EnvType.SINGULAR

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space
    
    def get_num_envs(self):
        return self.num_envs
