
try:
    import minigrid, miniworld
    from miniworld.wrappers import PyTorchObsWrapper, GreyscaleWrapper
except Exception:
    ...

import gymnasium as gym
import ale_py

import numpy as np

from .environment import Environment
from .wrappers import SpecifyActions, SwapChannel

import utils

class Gymenv(Environment):
    def __init__(self, env_name: str, num_envs: int, 
                 seed: int = None, normalise_obs: bool = False, 
                 atari: bool = False,  
                 allowed_actions: list = None,
                 swap_channel: bool = False,
                 **env_kwargs):
        
        self.env_name = env_name
        self.num_envs = num_envs
        self.seed = seed
        self.normalise_obs = normalise_obs
        self.allowed_actions = allowed_actions
        self.atari = "ALE" in env_name or "Pong" in env_name or "Boxing" in env_name or "SpaceInvaders" in env_name or "Breakout" in env_name
        self.minigrid = "MiniGrid" in env_name
        self.miniworld = "MiniWorld" in env_name

        def make_one_env():
            env = gym.make(self.env_name, **env_kwargs)

            if self.atari:
                env = gym.make(self.env_name, frameskip=1, **env_kwargs)
                env = gym.wrappers.AtariPreprocessing(env,
                    noop_max=30, frame_skip=4, terminal_on_life_loss=False,
                    screen_size=84, grayscale_obs=True, grayscale_newaxis=False
                )
                env = gym.wrappers.FrameStackObservation(env, stack_size=4)
                # env = gym.wrappers.ClipReward(env, min_reward=-1, max_reward=1)

            if self.minigrid:
                env = minigrid.wrappers.ImgObsWrapper(env)

            if self.miniworld:
                # env = GreyscaleWrapper(env)
                env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
                env = PyTorchObsWrapper(env)

            if self.normalise_obs:
                env = gym.wrappers.NormalizeObservation(env)

            if swap_channel:
                env = SwapChannel(env)

            if self.allowed_actions is not None:
                env = SpecifyActions(env, self.allowed_actions)

            return env
        
        list_of_envs = [make_one_env for env_idx in range(self.num_envs)]
        self.env = gym.vector.SyncVectorEnv(list_of_envs)
        self.env = gym.wrappers.vector.RecordEpisodeStatistics(self.env)

        self.single_state_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space

        self.start_states, self.start_info = self.env.reset(seed=self.seed)
    
    def get_normalised_obs(self):
        normalised_data = []
        for env in self.env.env.envs:
            normalise_wrapper = utils.get_wrapper(env, gym.wrappers.NormalizeObservation)
            if normalise_wrapper is not None:
                normalised_data.append({
                    "mean": normalise_wrapper.obs_rms.mean.copy(), 
                    "var": normalise_wrapper.obs_rms.var.copy(),
                    "count": normalise_wrapper.obs_rms.count,
                })
        return normalised_data
    
    def load_normalised_obs(self, normalised_data):
        if self.normalise_obs:
            for env_idx, env in enumerate(self.env.env.envs):
                normalise_wrapper = utils.get_wrapper(env, gym.wrappers.NormalizeObservation)
                if normalise_wrapper is not None:
                    normalise_wrapper.obs_rms.mean = normalised_data[env_idx]["mean"]
                    normalise_wrapper.obs_rms.var = normalised_data[env_idx]["var"]
                    normalise_wrapper.obs_rms.count = normalised_data[env_idx]["count"]

    def step(self, actions: np.array):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated, info
    
    def get_num_envs(self):
        return self.num_envs
    
    def get_single_state_space(self):
        return self.single_state_space
    
    def get_single_action_space(self):
        return self.single_action_space
    
    def get_start_states(self):
        return self.start_states
    
    def is_conv(self):
        return self.atari or self.minigrid or self.miniworld