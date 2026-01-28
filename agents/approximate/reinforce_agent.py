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
        self.fc1 = nn.Linear(state_space_dim, 8)
        self.fc2 = nn.Linear(8, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        output = F.softmax(self.fc2(f1))
        return output

class ReinforceAgent(Agent):
    def __init__(self, lr, gamma, normalise, time_limit=10000):
        self.lr = lr
        self.gamma = gamma
        self.normalise = normalise
        self.time_limit = time_limit

    def normalise_state(self, s):
        s = torch.from_numpy(s).float().clone()
        if self.normalise:
            for i in range(len(self.state_space_mins)):
                s[i] = (s[i] - self.state_space_mins[i]) / (self.state_space_maxs[i] - self.state_space_mins[i])
        return s

    def run_policy(self, s, t):
        with torch.no_grad():
            probs = self.nn.forward(self.normalise_state(s)).numpy()
        # print(probs)
        return np.argmax(probs)
        
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
            self.nn = NN(self.state_space_size, self.action_space_size)
            self.optimiser = optim.Adam(self.nn.parameters(), lr=self.lr)
            self.reward_history = []

    def finish_episode(self):
        
        G = 0
        for t in range(len(self.steps)-1, -1, -1):
            s, sprime, a, r = self.steps[t]
            G = r + self.gamma * G

            self.optimiser.zero_grad()
            loss = G * torch.log(self.nn.forward(self.normalise_state(s))[a])
            loss.backward()
            self.optimiser.step()

        self.steps = []
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
