import os, warnings

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class StateValueNN(nn.Module):
    def __init__(self, state_space_dim):
        super(StateValueNN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output

class PolicyNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = F.log_softmax(self.fc3(f2))
        return output

class ReinforceBaselineAgent(Agent):
    def __init__(self, policy_lr, state_value_lr, gamma, normalise, time_limit=10000):
        self.policy_lr = policy_lr
        self.state_value_lr = state_value_lr
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
            probs = self.policy_nn.forward(self.normalise_state(s)).numpy()
        probs = np.exp(probs)
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
            self.policy_nn = PolicyNN(self.state_space_size, self.action_space_size)
            self.policy_optimiser = optim.Adam(self.policy_nn.parameters(), lr=self.policy_lr)
            self.state_value_nn = StateValueNN(self.state_space_size)
            self.state_value_optimiser = optim.Adam(self.state_value_nn.parameters(), lr=self.state_value_lr)
            self.reward_history = []

    def finish_episode(self):

        for t in range(0, len(self.steps)-1):
            s, sprime, a, r = self.steps[t]

            self.policy_optimiser.zero_grad()
            self.state_value_optimiser.zero_grad()

            G = 0
            for k in range(t+1, len(self.steps)):
                _, _, _, rk = self.steps[k]
                G += (self.gamma**(k-t-1)) * rk

           
            td_err = G - self.state_value_nn.forward(self.normalise_state(s))

            # state_value_loss = td_err * self.state_value_nn.forward(self.normalise_state(s))
            state_value_loss = 0.5 * td_err.pow(2)
            state_value_loss.backward(retain_graph=True)

            nn_loss = (self.gamma**t) * td_err * self.policy_nn.forward(self.normalise_state(s))[a]
            nn_loss.backward()

            self.policy_optimiser.step()
            self.state_value_optimiser.step()

        self.steps = []
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]

