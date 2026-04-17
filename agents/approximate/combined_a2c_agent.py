from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CombinedNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, conv):
        super(CombinedNN, self).__init__()
        self.conv = conv
        fc_input_dim = 3136 if conv else state_space_dim[0]

        if self.conv:
            self.conv_nn = nn.Sequential(
                nn.Conv2d(state_space_dim[0], 32, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.Flatten() # (3136,)
            )

        self.fc_nn = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.policy_nn = nn.Sequential(
            nn.Linear(64, action_space_dim),
            nn.LogSoftmax(dim=1)
        )

        self.state_value_nn = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, input):
        if self.conv:
            new_input = input / 255.0
            conv_out = self.conv_nn(new_input)
            fc_out = self.fc_nn(conv_out)
            return self.policy_nn(fc_out), self.state_value_nn(fc_out)
        else:
            fc_out = self.fc_nn(input)
            return self.policy_nn(fc_out), self.state_value_nn(fc_out)

class CombinedA2CAgent(Agent):
    def __init__(self, device, writer, lr, gamma, conv,
                 tmax, entropy_weight=0.01, value_weight=1.0, clip_grad_norm=None,
                 save_path=None, load_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.gamma = gamma
        self.conv = conv
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.clip_grad_norm = clip_grad_norm
        self.tmax = tmax
        self.eval = False
        self.save_path = save_path
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        with torch.no_grad():
            probs, _ = self.combined_nn(self.process_state(s)) # (num_envs, 6)
            probs = probs.cpu().numpy() # (num_envs, 6)
        probs = np.exp(probs) # (num_envs, 6)
        actions = np.array([
            np.random.choice(self.action_space_size, p=probs[i]) for i in range(probs.shape[0])
        ]) # (num_envs,)
        return actions

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs

        self.total_updates_made = 0
        self.time_steps = np.zeros((self.num_envs,), dtype=np.int32)
        self.transitions = [[] for _ in range(self.num_envs)]
        self.reward_history = []
        self.current_episode_rewards = np.zeros((self.num_envs,))

        self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size, self.conv).to(self.device)
        self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.combined_nn.load_state_dict(checkpoint["nn"])
            self.combined_optimiser.load_state_dict(checkpoint["optimiser"])

    def make_a2c_update(self, env_idx):
        self.total_updates_made += 1

        # select and unpack transitions
        env_transitions = self.transitions[env_idx]
        all_s = torch.tensor(np.array([t[0] for t in env_transitions]), dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        all_a = torch.tensor(np.array([t[1] for t in env_transitions]), dtype=torch.int64).to(self.device) # (tmax,)
        all_r = torch.tensor(np.array([t[2] for t in env_transitions]), dtype=torch.float32).to(self.device) # (tmax,)
        all_sprime = torch.tensor(np.array([t[3] for t in env_transitions]), dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        all_done = torch.tensor(np.array([t[4] for t in env_transitions]), dtype=torch.float32).to(self.device) # (tmax,)
        is_terminal = all_done[-1]

        # containers for loss
        all_policy_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_state_value_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_entropy_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar

        # initialise R according to whether last transition was terminal
        if is_terminal:
            # R = 0 for terminal s_t
            R = torch.tensor([0.0], dtype=torch.float32, requires_grad=True).to(self.device) # scalar
        else:
            # R = V(s_t) for non-terminal s_t
            _, last_state_bootstrap = self.combined_nn(all_sprime[-1].unsqueeze(0)) # (1, action_space_dim,), (1, 1)
            R = last_state_bootstrap.squeeze(1) # scalar
            
        # calculate R and accumulate policy and state value gradients
        for t in reversed(range(len(env_transitions))):
            R = all_r[t] + self.gamma * R * (1 - all_done[t]) # scalar

            policy_prediction, state_value_prediction = self.combined_nn(all_s[t].unsqueeze(0)) # (1, action_space_dim), (1, 1)
            with torch.no_grad():
                advantage = R - state_value_prediction.squeeze(1) # scalar
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # normalise advantage
            
            log_prob = policy_prediction.squeeze(0)[all_a[t]] # scalar

            all_policy_loss_total += -advantage.item() * log_prob # scalar
            all_state_value_loss_total += F.mse_loss(state_value_prediction.squeeze(1), R)
            all_entropy_total += -(torch.exp(policy_prediction) * policy_prediction).sum() # scalar
        
        combined_loss = (
            all_policy_loss_total
            + (self.value_weight * all_state_value_loss_total)
            - (self.entropy_weight * all_entropy_total)
        ) # scalar (loss)

        if self.writer is not None:
            self.writer.add_scalar("policy_loss", all_policy_loss_total.item(), self.total_updates_made)
            self.writer.add_scalar("state_value_loss", all_state_value_loss_total.item(), self.total_updates_made)
            self.writer.add_scalar("combined_loss", combined_loss.item(), self.total_updates_made)

        # backprop combined NN
        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()

    def update(self, s, sprime, a, r, done):

        # s = (num_envs, 4, 84, 84)
        # sprime = (num_envs, 4, 84, 84)
        # a = (num_envs,)
        # r = (num_envs,)
        # done = (num_envs,)

        self.current_episode_rewards += r
        self.time_steps += 1

        for env_idx in range(self.num_envs):
            self.transitions[env_idx].append((s[env_idx], a[env_idx], r[env_idx], sprime[env_idx], done[env_idx]))

            # if tmax steps are reached or episode terminates, make A2C update
            if self.time_steps[env_idx] % self.tmax == 0 or done[env_idx]:
                self.make_a2c_update(env_idx)
                self.transitions[env_idx] = []
                
            if done[env_idx]:
                # if terminal, save/log/reset rewards, save model
                self.reward_history.append(self.current_episode_rewards[env_idx])

                if self.writer is not None:
                    self.writer.add_scalar("mean_episode_reward", np.mean(self.reward_history[-100:]), len(self.reward_history))
                    self.writer.add_scalar("episode_reward", self.current_episode_rewards[env_idx], len(self.reward_history))
                self.current_episode_rewards[env_idx] = 0.0

                if self.save_path is not None:
                    torch.save({
                        "nn": self.combined_nn.state_dict(),
                        "optimiser": self.combined_optimiser.state_dict(),
                    }, self.save_path)

    def finish_episode(self, episode_num):
        pass

    def toggle_eval(self):
        self.eval = not self.eval

    def get_supported_env_types(self):
        return [EnvType.SINGULAR, EnvType.VECTORISED]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]