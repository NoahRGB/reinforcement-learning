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
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output

class SemigradientSarsaAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, decay_rate=1.0, time_limit=10000):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def normalise_state(self, s):
        s = torch.from_numpy(s).float().clone()
        s[0] = (s[0] - (-1.2)) / (0.6 - (-1.2))
        s[1] = (s[1] - (-0.07)) / (0.07 - (-0.07))
        return s 

    def initialise(self, state_space, action_space, start_state, resume=False):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions[0] 
        self.action_space_size = action_space.dimensions 
        if not resume:
            self.nn = NN(self.state_space_size, self.action_space_size)
            self.optimiser = optim.Adam(self.nn.parameters(), lr=0.001)
            # self.optimiser = optim.SGD(self.nn.parameters(), lr=0.00001)
            self.reward_history = []
        self.action = self.generate_action(start_state)
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.action = self.generate_action(self.start_state)
        self.epsilon *= self.decay_rate

    def get_all_actions(self):
        return [i for i in range(self.action_space_size)]

    def get_best_actions(self, s):
        with torch.no_grad():
            qvals = self.nn.forward(self.normalise_state(s)).numpy()
        return np.where(qvals == qvals.max())[0]

    def run_policy(self, s, t):
        return self.action

    def generate_action(self, s):
        if np.random.random() >= self.epsilon:
            return np.random.choice(self.get_best_actions(s))
        return np.random.choice(self.get_all_actions())

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r

        aprime = self.generate_action(sprime) 

        self.optimiser.zero_grad()

        qs = self.nn.forward(self.normalise_state(s))
        qsa = qs[a]

        if done:
            target = torch.tensor(r)
            # print(qs)
        else:
            with torch.no_grad():
                target = r + self.gamma * self.nn.forward(self.normalise_state(sprime))[aprime]

        td_err = target - qsa
        loss = F.mse_loss(qsa, target.detach())
        # loss = 0.5 * td_err.pow(2)
        # loss = -td_err * qsa
        loss.backward()
        self.optimiser.step()

        self.action = aprime

        self.time_step += 1
        return self.time_step >= self.time_limit

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]

