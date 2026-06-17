import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class QNet(torch.nn.Module):
    def __init__(self, input_size, output_size, conv):
        super(QNet, self).__init__()
        self.conv = conv

        if conv:
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(3136, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, output_size)
            )
        else:
            self.body = torch.nn.Sequential(
                torch.nn.Linear(*input_size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, output_size),
            )
    
    def forward(self, inp):
        if self.conv:
            new_inp = inp / 255.0
            return self.body(new_inp)
        return self.body(inp)

class DoubleDQN(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, gamma, epsilon_scheduler, cgn, warmup_steps, gradient_steps):
        self.lr = lr
        self.replay_size = replay_size
        self.C = C
        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = epsilon_scheduler.get_value()
        self.cgn = cgn
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.device = torch.device("cpu")

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())

        self.replay = deque(maxlen=self.replay_size)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)

        self._update_target_net()

        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            if np.random.random() >= self.epsilon:
                q_values = self.qnet(states)
                actions = q_values.argmax(dim=-1)
                return actions
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64).to(self.device)
        
    def _improve(self):
        if len(self.replay) < self.minibatch_size: return

        minibatch = random.sample(self.replay, self.minibatch_size)
        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch)
        
        all_s = torch.cat(all_s).to(self.device)
        all_a = torch.cat(all_a).to(self.device)
        all_r = torch.cat(all_r).to(self.device)
        all_sprime = torch.cat(all_sprime).to(self.device)
        all_done = torch.cat(all_done).to(self.device)

        q_vals = self.qnet(all_s) # (minibatch_size, action_space_dim,)
        chosen_q_vals = q_vals.gather(1, all_a.unsqueeze(1)).squeeze(1) # (minibatch_size,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            greedy_action_selection = self.qnet(all_sprime).argmax(dim=-1) # (minibatch_size,)
            greedy_action_evaluation = self.target_qnet(all_sprime).gather(-1, greedy_action_selection.unsqueeze(1)).squeeze(1)
            targets = all_r + self.gamma * greedy_action_evaluation * (1 - all_done)

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(chosen_q_vals, targets) # scalar
        loss.backward()
        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cgn)
        self.optim.step()

        self.logger.gradient_step_complete(["qnet_loss"], [loss.item()])
        self.logger.network_update({"qnet":self.qnet.state_dict(), "target_qnet":self.target_qnet.state_dict(), "optim":self.optim.state_dict()})


    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None):
        assert env.num_envs == 1
        assert utils.is_space_discrete(env.get_single_action_space())

        utils.seed(seed)
        self.logger = logger
        total_iterations = total_timesteps // self.update_freq
        current_game_states = torch.from_numpy(env.start_states).float().to(self.device)

        self._setup(env)

        for iteration in range(1, total_iterations + 1):

            for current_t in range(self.update_freq):
                self.logger.timestep_complete()

                if self.logger.timesteps_completed % self.C == 0:
                    self._update_target_net()
                self.epsilon = self.epsilon_scheduler.step()
            
                current_actions = self._get_actions(current_game_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)
                        
                current_rewards = torch.from_numpy(current_rewards).float().to(self.device)
                current_sprimes = torch.from_numpy(current_sprimes).float().to(self.device)
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float().to(self.device)

                self.replay.append((
                    current_game_states.detach(),
                    current_actions.detach(),
                    current_rewards,
                    current_sprimes,
                    current_dones,
                ))

                current_game_states = current_sprimes
    
                if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve()
            
            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        