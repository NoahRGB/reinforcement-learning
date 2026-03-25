from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CombinedNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(CombinedNN, self).__init__()

        self.conv_nn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU()
        )

        self.policy_nn = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_dim),
            nn.LogSoftmax(dim=1)
        )

        self.state_value_nn = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        input = input / 255.0
        conv_out = self.conv_nn(input)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.policy_nn(conv_out), self.state_value_nn(conv_out)

class ConvA2CAgent(Agent):
    def __init__(self, device, writer, lr, gamma, tmax,
                 save_nn_path=None, load_nn_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.tmax = tmax
        self.eval = False
        self.gamma = gamma
        self.save_nn_path = save_nn_path
        self.load_nn_path = load_nn_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        with torch.no_grad():
            probs, _ = self.combined_nn(torch.stack([self.process_state(s)]))
            probs = probs.cpu().numpy()
        probs = np.exp(probs).squeeze(0)
        return np.random.choice([i for i in range(self.action_space_size)], p=probs)

    def initialise(self, state_space, action_space, start_state, resume=False):
        self.transitions = []
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.state_space_mins = state_space.min_bound
        self.state_space_maxs = state_space.max_bound
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.reward_history = []

            self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size).to(self.device)
            self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)

            # load saved models
            if self.load_nn_path != None:
                self.combined_nn.load_state_dict(torch.load(self.load_nn_path))

    def make_update(self):
        torch.autograd.set_detect_anomaly(True, check_nan=False)
        final_s, _, _, _, terminal = self.transitions[-1]

        R = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        state_value_predictions = torch.tensor([0.0], dtype=torch.float32).to(self.device) 
        policy_loss_total = torch.tensor([0.0], dtype=torch.float32).to(self.device) 

        if not terminal:
            R = self.combined_nn(torch.stack([self.process_state(final_s)]))[1].squeeze().detach()

        for i in reversed(range(len(self.transitions))):
            (s, sprime, a, r, done)  = self.transitions[i]
            R = r + self.gamma * R

            policy_prediction, state_value_prediction = self.combined_nn(torch.stack([self.process_state(s)]))
            policy_prediction = policy_prediction.squeeze(0) # remove the batch dim
            state_value_predictions =  state_value_predictions + state_value_prediction.squeeze(0) # remove the batch dim

            advantage = R - state_value_prediction
            policy_loss_total = policy_loss_total - (advantage * policy_prediction[a])

        state_value_loss = F.mse_loss(state_value_predictions, R.unsqueeze(0))
        combined_loss = policy_loss_total + state_value_loss

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        self.combined_optimiser.step()

        self.transitions = []


    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r

        self.transitions.append((s, sprime, a, r, done))

        if done or self.time_step % self.tmax == 0:
            self.make_update()

        self.time_step += 1


    def finish_episode(self, episode_num):
        self.transitions = []
        self.reward_history.append(self.current_episode_rewards)

        if self.writer != None:
            self.writer.add_scalar("episode_reward", self.current_episode_rewards, episode_num)
        self.current_episode_rewards = 0

        # save models
        if self.save_nn_path != None:
            torch.save(self.combined_nn.state_dict(), self.save_nn_path)

    def toggle_eval(self):
        self.eval = not self.eval

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]

