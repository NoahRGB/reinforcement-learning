import os, warnings

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class NN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_space_dim)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = F.log_softmax(output, dim=0)
        return output

class ReinforceAgent(Agent):
    def __init__(self, device, writer, lr, gamma, normalise, time_limit=10000):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.gamma = gamma
        self.normalise = normalise
        self.time_limit = time_limit

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        with torch.no_grad():
            probs = self.nn.forward(self.process_state(s)).cpu().numpy()
        probs = np.exp(probs) # exp() because output is LOG_softmax (is there a better way to do this?)
        return np.random.choice([i for i in range(self.action_space_size)], p=probs)
        
    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r
        self.steps.append((s, sprime, a, r))
        self.time_step += 1
        return self.time_step >= self.time_limit

    def initialise(self, state_space, action_space, start_state, resume=False):
        self.steps = []
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.state_space_mins = state_space.min_bound
        self.state_space_maxs = state_space.max_bound
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.nn = NN(self.state_space_size, self.action_space_size).to(self.device)
            self.optimiser = optim.Adam(self.nn.parameters(), lr=self.lr)
            self.reward_history = []

    def finish_episode(self, episode_num):
        
        self.optimiser.zero_grad()

        G = 0
        loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) 

        # accumulate loss
        for t in range(len(self.steps)-1, -1, -1):
            s, sprime, a, r = self.steps[t]
            G = self.gamma * G + r
            # note this is negated since default is to do gradient DESCENT but policy gradient
            # wants gradient ASCENT on performance measure J
            loss_total += -(self.gamma**t) * (G - 0) * self.nn.forward(self.process_state(s))[a]

        self.writer.add_scalar("policy_loss", loss_total.item(), episode_num)
        loss_total.backward()
        self.optimiser.step()

        self.steps = []
        self.reward_history.append(self.current_episode_rewards)
        self.writer.add_scalar("episode_reward", self.current_episode_rewards, episode_num)
        self.current_episode_rewards = 0

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
