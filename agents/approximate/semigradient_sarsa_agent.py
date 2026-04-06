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

class SemigradientSarsaAgent(Agent):
    def __init__(self, device, writer, normalise, lr, epsilon, gamma, decay_rate=1.0, 
                 load_nn_path=None, save_nn_path=None):
        self.device = device
        self.writer = writer
        self.normalise = normalise
        self.lr = lr
        self.eval = False
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.load_nn_path = load_nn_path
        self.save_nn_path = save_nn_path

    def normalise_state(self, s):
        s = torch.from_numpy(s).float().clone()
        if self.normalise:
            for i in range(len(self.state_space_mins)):
                s[i] = (s[i] - self.state_space_mins[i]) / (self.state_space_maxs[i] - self.state_space_mins[i])
        return s

    def initialise(self, state_space, action_space, start_state, num_envs, resume=False):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions 
        self.action_space_size = action_space.dimensions 
        self.state_space_mins = state_space.min_bounds
        self.state_space_maxs = state_space.max_bounds
        self.num_envs = num_envs
        if not resume:
            print(self.state_space_size)
            self.nn = NN(self.state_space_size, self.action_space_size)
            if self.load_nn_path != None:
                self.nn.load_state_dict(torch.load(self.load_nn_path))
            self.optimiser = optim.Adam(self.nn.parameters(), lr=self.lr)
            self.reward_history = []
        self.action = self.generate_action(start_state)
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self, episode_num):
        self.reward_history.append(self.current_episode_rewards)
        self.action = self.generate_action(self.start_state)
        if self.writer != None:
            self.writer.add_scalar("episode_reward", self.current_episode_rewards, episode_num)
        self.current_episode_rewards = 0
        self.epsilon *= self.decay_rate

    def get_all_actions(self):
        return [i for i in range(self.action_space_size)]

    def get_best_actions(self, s):
        with torch.no_grad():
            qvals = self.nn.forward(self.normalise_state(s)).numpy()
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

        qs = self.nn.forward(self.normalise_state(s[0]))
        qsa = qs[a]

        if done[0]:
            target = torch.tensor(r[0])
        else:
            with torch.no_grad():
                target = r[0] + self.gamma * self.nn.forward(self.normalise_state(sprime[0]))[aprime]

        td_err = target - qsa

        loss = 0.5 * td_err.pow(2)
        # loss = -td_err * qsa
        loss.backward()
        self.optimiser.step()

        self.action = aprime

        if done[0] and self.save_nn_path != None:
            torch.save(self.nn.state_dict(), self.save_nn_path)

        self.time_step += 1

    def toggle_eval(self):
        if not self.eval:
            self.epsilon_checkpoint = self.epsilon
            self.epsilon = 0.0
        else:
            self.epsilon = self.epsilon_checkpoint
        self.eval = not self.eval

    def get_supported_env_types(self):
        return [EnvType.SINGULAR]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace, ContinuousSpace]

