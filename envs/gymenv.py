import minigrid
import gymnasium as gym
import ale_py

import numpy as np

from .environment import Environment

class Gymenv(Environment):
    def __init__(self, env_name: str, num_envs: int, seed: int = None, normalise_obs: bool = False, atari: bool = False, **env_kwargs):
        self.env_name = env_name
        self.num_envs = num_envs
        self.seed = seed
        self.normalise_obs = normalise_obs
        self.atari = "ALE" in env_name or "Pong" in env_name
        self.minigrid = "MiniGrid" in env_name

        def make_one_env():
            if self.atari:
                env = gym.make(self.env_name, **env_kwargs)
                env = gym.wrappers.AtariPreprocessing(env,
                    noop_max=30, frame_skip=4, terminal_on_life_loss=False,
                    screen_size=84, grayscale_obs=True, grayscale_newaxis=False
                )
                env = gym.wrappers.FrameStackObservation(env, stack_size=2)
            elif self.minigrid:
                env = gym.make(self.env_name, **env_kwargs)
                env = minigrid.wrappers.ImgObsWrapper(env)
                env = gym.wrappers.FlattenObservation(env)
            else:
                env = gym.make(self.env_name, **env_kwargs)

            if self.normalise_obs:
                env = gym.wrappers.NormalizeObservation(env)

            return env
        
        list_of_envs = [make_one_env for env_idx in range(self.num_envs)]
        self.env = gym.vector.SyncVectorEnv(list_of_envs)
        self.env = gym.wrappers.vector.RecordEpisodeStatistics(self.env)

        self.single_state_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space

        self.start_states, self.start_info = self.env.reset(seed=self.seed)

    def step(self, actions: np.array):
        return self.env.step(actions)
    
    def get_num_envs(self):
        return self.num_envs
    
    def get_single_state_space(self):
        return self.single_state_space
    
    def get_single_action_space(self):
        return self.single_action_space
    
    def get_start_states(self):
        return self.start_states
    
    def is_conv(self):
        return self.atari