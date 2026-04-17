from environments.environment import Environment
from environments.spaces import detect_space, EnvType

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, GrayscaleObservation, ResizeObservation, ReshapeObservation

import numpy as np

class GymEnvironment(Environment):
    def __init__(self, name, num_envs, image_preprocess=False, **kwargs):
        
        self.num_envs = num_envs

        def make_one_env():
            env = gym.make(name, **kwargs)
            if image_preprocess:
                env = GrayscaleObservation(env, keep_dim=True)
                env = ResizeObservation(env, (84, 84))
                env = ReshapeObservation(env, (1, 84, 84))
            return env

        if self.num_envs == 1:
            # create a single environment
            self.env = make_one_env()
            self.state_space = detect_space(self.env.observation_space)
            self.action_space = detect_space(self.env.action_space)
        else:
            # create multiple environments in a sync vector
            envs = [make_one_env for _ in range(self.num_envs)]
            self.env = gym.vector.SyncVectorEnv(envs)
            self.state_space = detect_space(self.env.single_observation_space)
            self.action_space = detect_space(self.env.single_action_space)

    def __del__(self):
        self.env.close()

    def step(self, s, a):
        if self.num_envs == 1:
            # if using a singular environment, add a fake dimension
            sprime, r, is_terminated, is_truncated, info = self.env.step(a[0])
            return np.expand_dims(sprime, axis=0), np.array([r]), np.array([is_terminated or is_truncated])
        else:
            sprime, r, is_terminated, is_truncated, info = self.env.step(a)
            return sprime, r, (is_terminated|is_truncated)

    def reset(self):
        self.env.reset()

    def get_start_state(self):
        start_state, _ = self.env.reset()
        if self.num_envs == 1:
            # add a fake dimension to represent the 1 environment
            return np.expand_dims(start_state, axis=0)
        return start_state
    
    def get_env_type(self):
        return EnvType.SINGULAR if self.num_envs == 1 else EnvType.VECTORISED

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space
    
    def get_num_envs(self):
        return self.num_envs
