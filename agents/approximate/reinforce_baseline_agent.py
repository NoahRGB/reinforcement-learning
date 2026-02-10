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
        self.fc1 = nn.Linear(state_space_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output

class PolicyNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = F.log_softmax(self.fc3(f2), dim=0)
        return output

class ReinforceBaselineAgent(Agent):
    def __init__(self, device, writer, policy_lr, state_value_lr, gamma, normalise, save_policy_nn_path=None, save_state_value_nn_path=None, load_policy_nn_path=None, load_state_value_nn_path=None, time_limit=10000):
        self.device = device
        self.writer = writer
        self.policy_lr = policy_lr
        self.state_value_lr = state_value_lr
        self.gamma = gamma
        self.normalise = normalise
        self.save_policy_nn_path = save_policy_nn_path
        self.save_state_value_nn_path = save_state_value_nn_path
        self.load_policy_nn_path = load_policy_nn_path
        self.load_state_value_nn_path = load_state_value_nn_path
        self.time_limit = time_limit

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        with torch.no_grad():
            probs = self.policy_nn(self.process_state(s)).cpu().numpy()
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
            self.policy_nn = PolicyNN(self.state_space_size, self.action_space_size).to(self.device)
            self.policy_optimiser = optim.Adam(self.policy_nn.parameters(), lr=self.policy_lr)
            self.state_value_nn = StateValueNN(self.state_space_size).to(self.device)
            self.state_value_optimiser = optim.Adam(self.state_value_nn.parameters(), lr=self.state_value_lr)
            if self.load_policy_nn_path != None:
                self.policy_nn.load_state_dict(torch.load(self.load_policy_nn_path))
            if self.load_state_value_nn_path != None:
                self.state_value_nn.load_state_dict(torch.load(self.load_state_value_nn_path))
            self.reward_history = []

    def finish_episode(self, episode_num):

        self.policy_optimiser.zero_grad()
        self.state_value_optimiser.zero_grad()

        G = 0
        state_value_predictions = torch.tensor([0.0], dtype=torch.float32).to(self.device) 
        policy_loss_total = torch.tensor([0.0], dtype=torch.float32).to(self.device) 

        # accumulate losses
        for t in range(len(self.steps)-1, -1, -1):
            s, sprime, a, r = self.steps[t]
            G = self.gamma * G + r

            state_value_prediction = self.state_value_nn(self.process_state(s))
            state_value_predictions = state_value_predictions + state_value_prediction
            policy_loss_total = policy_loss_total + -(self.gamma**t) * (G - state_value_prediction) * self.policy_nn(self.process_state(s))[a]

        self.writer.add_scalar("policy_loss", policy_loss_total.item(), episode_num)
        policy_loss_total.backward(retain_graph=True)
        self.policy_optimiser.step()

        targets = torch.tensor([G], dtype=torch.float32).to(self.device)
        state_value_loss = F.mse_loss(state_value_predictions, targets)
        self.writer.add_scalar("sv_loss", state_value_loss.item(), episode_num)
        state_value_loss.backward()
        self.state_value_optimiser.step()

        self.steps = []
        self.reward_history.append(self.current_episode_rewards)
        self.writer.add_scalar("episode_reward", self.current_episode_rewards, episode_num)
        self.writer.flush()
        self.current_episode_rewards = 0

        if self.save_policy_nn_path != None:
            torch.save(self.policy_nn.state_dict(), self.save_policy_nn_path)
        if self.save_state_value_nn_path != None:
            torch.save(self.state_value_nn.state_dict(), self.save_state_value_nn_path)


    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]

