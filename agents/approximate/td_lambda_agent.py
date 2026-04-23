from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class NN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(*state_space_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output

class TDLambdaAgent(Agent):
    def __init__(self, lambd, alpha, epsilon, gamma, decay_rate=1.0):
        self.lambd = lambd
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions 
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs
        
        self.nn = NN(self.state_space_size, self.action_space_size)
        self.optimiser = optim.Adam(self.nn.parameters(), lr=self.alpha)
        self.z = {param: torch.zeros_like(param) for param in self.nn.parameters()}
        self.reward_history = []

        self.action = self.generate_action(start_state)
        self.termination_time = np.inf
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self, episode_num):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.action = self.generate_action(self.start_state[0])
        self.epsilon *= self.decay_rate
        self.z = {param: torch.zeros_like(param) for param in self.nn.parameters()}

    def prepare_state(self, s):
        return torch.from_numpy(s).float().clone()

    def get_all_actions(self):
        return [i for i in range(self.action_space_size)]

    def get_best_actions(self, s):
        with torch.no_grad():
            qvals = self.nn(self.prepare_state(s)).numpy()
        best = np.where(qvals == qvals.max())[0]
        return best 

    def run_policy(self, s, t):
        return self.action

    def generate_action(self, s):
        if np.random.random() >= self.epsilon:
            return np.random.choice(self.get_best_actions(s))
        return np.random.choice(self.get_all_actions())

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r[0]

        aprime = self.generate_action(sprime[0]) 

        self.optimiser.zero_grad()
        # self.nn.zero_grad()

        qs = self.nn(self.prepare_state(s[0]))
        qsa = qs[a]

        if done:
            target = torch.tensor(r[0])
        else:
            with torch.no_grad():
                target = r[0] + self.gamma * self.nn(self.prepare_state(sprime[0]))[aprime]

        td_err = target - qsa

        # qsa.backward()
        loss = 0.5 * td_err.pow(2)
        loss.backward()

        for param in self.nn.parameters():
            self.z[param] = self.gamma * self.lambd * self.z[param] + param.grad
            param.grad.copy_(td_err * self.z[param])

  
        self.optimiser.step()

        self.action = aprime

        self.time_step += 1

    def get_supported_env_types(self):
        return [EnvType.SINGULAR]

    def get_supported_state_spaces(self):
        return [DiscreteSpace, ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
