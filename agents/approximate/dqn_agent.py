from collections import deque
import random

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output

class DQNAgent(Agent):
    def __init__(self, device, writer, lr, replay_memory_size, minibatch_size, epsilon, gamma, decay_rate=1.0, load_nn_path=None, save_nn_path=None, time_limit=10000):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.time_limit = time_limit
        self.load_nn_path = load_nn_path
        self.save_nn_path = save_nn_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def initialise(self, state_space, action_space, start_state, resume=False):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions 
        self.action_space_size = action_space.dimensions 
        self.state_space_mins = state_space.min_bound
        self.state_space_maxs = state_space.max_bound
        self.actions = [i for i in range(self.action_space_size)]
        if not resume:
            self.dqn = DQN(self.state_space_size, self.action_space_size).to(self.device)
            if self.load_nn_path != None:
                self.dqn.load_state_dict(torch.load(self.load_nn_path))
            self.optimiser = optim.Adam(self.dqn.parameters(), lr=self.lr)
            self.replay = deque(maxlen=self.replay_memory_size)
            self.reward_history = []
        self.action = self.generate_action(start_state)
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self, episode_num):
        self.reward_history.append(self.current_episode_rewards)
        self.writer.add_scalar("episode_reward", self.current_episode_rewards, episode_num)
        self.writer.flush()
        self.current_episode_rewards = 0
        self.action = self.generate_action(self.start_state)
        self.epsilon *= self.decay_rate

    def get_best_actions(self, s):
        with torch.no_grad():
            qvals = self.dqn.forward(self.process_state(s)).cpu().numpy()
        return np.where(qvals == qvals.max())[0]

    def run_policy(self, s, t):
        return self.action

    def generate_action(self, s):
        if np.random.random() >= self.epsilon:
            return np.random.choice(self.get_best_actions(s))
        return np.random.choice(self.actions)

    def replay_memory_update(self):
        minibatch = random.sample(self.replay, self.minibatch_size)
        
        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch)
        all_s = torch.stack([self.process_state(s_) for s_ in all_s]).to(self.device)
        all_a = torch.tensor(all_a, dtype=torch.int32).to(self.device)
        all_r = torch.tensor(all_r, dtype=torch.float32).to(self.device)
        all_sprime = torch.stack([self.process_state(s_) for s_ in all_sprime]).to(self.device)
        all_done = torch.tensor(all_done, dtype=torch.float32).to(self.device)

        chosen_q_vals = self.dqn(all_s).gather(1, all_a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            targets = all_r + self.gamma * self.dqn(all_sprime).max(1)[0] * (1 - all_done)

        self.optimiser.zero_grad()
        loss = F.mse_loss(chosen_q_vals, targets)
        self.writer.add_scalar("loss", loss.item(), len(self.reward_history) + self.time_step)
        loss.backward()
        self.optimiser.step()

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r

        aprime = self.generate_action(sprime) 

        self.replay.append((s, a, r, sprime, done))
        self.writer.add_scalar("replay_size", len(self.replay), len(self.reward_history) + self.time_step)

        if len(self.replay) >= self.minibatch_size:
            self.replay_memory_update()

        self.action = aprime

        if done and self.save_nn_path != None:
            torch.save(self.dqn.state_dict(), self.save_nn_path)
        self.time_step += 1
        return self.time_step >= self.time_limit

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace, ContinuousSpace]
