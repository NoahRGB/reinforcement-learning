import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils


class NewSAC(agents.Agent):

    def __init__(self, lr, gamma, replay_size, minibatch_size, update_freq, 
                 alpha_start, auto_alpha, target_factor, warmup_steps, gradient_steps):
        self.lr = lr
        self.gamma = gamma
        self.replay_size = replay_size
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.alpha_start = alpha_start
        self.alpha = alpha_start
        self.target_factor = target_factor
        self.auto_alpha = auto_alpha
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.device = torch.device("cpu")

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            return 0.0

    
    def _setup(self, env: envs.Environment):
        self.logger.log_parameters(self)

    def _improve(self, env: envs.Environment):
        pass

    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None, quiet: bool = False):
        assert env.get_num_envs() == 1
        assert utils.is_space_continuous(env.get_single_action_space())

        total_iterations = total_timesteps // self.update_freq
        utils.seed(seed)
        self.logger = logger
        current_game_states = torch.from_numpy(env.get_start_states()).float().to(self.device)

        self._setup(env)

        for iteration in range(1, total_iterations + 1):
            
            for current_t in range(self.update_freq):
                self.logger.timestep_complete()

                current_actions = self._get_actions(current_game_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(self._scale_action(current_actions).cpu().numpy())

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)

                current_rewards = torch.from_numpy(current_rewards).float().to(self.device)
                current_sprimes = torch.from_numpy(current_sprimes).float().to(self.device)
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float().to(self.device)

                self.replay.append((
                    current_game_states,
                    current_actions,
                    current_rewards,
                    current_sprimes,
                    current_dones,
                ))

                current_game_states = current_sprimes

                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve(env)

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve(env)
        
        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device