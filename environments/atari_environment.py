from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation, ClipReward, RecordVideo
from environments.environment import Environment
from environments.spaces import EnvType, detect_space

import gymnasium as gym
import ale_py

import numpy as np

class AtariEnvironment(Environment):
    def __init__(self, name, **kwargs):

        self.env = gym.make(name, frameskip=1, **kwargs)
        self.env = AtariPreprocessing(self.env,
            noop_max=0, frame_skip=4, terminal_on_life_loss=False,
            screen_size=84, grayscale_obs=True, grayscale_newaxis=False
        )
        self.env = FrameStackObservation(self.env, stack_size=4)
        # self.env = ClipReward(self.env, -1, 1)

        # self.env = RecordVideo(self.env, ".", episode_trigger=lambda x: True)

        self.action_space = detect_space(self.env.action_space)
        self.state_space = detect_space(self.env.observation_space)
        self.num_envs = 1

    def __del__(self):
        self.env.close()

    def step(self, s, a):
        chosen_action = a
        if type(chosen_action) == np.ndarray:
            chosen_action = a[0]
        sprime, r, is_terminated, is_truncated, info = self.env.step(chosen_action)
        return np.expand_dims(sprime, 0), np.array([r]), np.array([is_terminated or is_truncated])

    def reset(self):
        self.env.reset()

    def get_start_state(self):
        start_state, _ = self.env.reset()
        return np.expand_dims(start_state, 0)
    
    def get_env_type(self):
        return EnvType.SINGULAR

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space
    
    def get_num_envs(self):
        return self.num_envs
