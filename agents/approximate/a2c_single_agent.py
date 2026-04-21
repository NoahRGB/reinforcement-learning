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

        self.non_conv_fc_nn = nn.Sequential(
            nn.Linear(state_space_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.conv_fc_nn = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
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
            fc_out = self.conv_fc_nn(conv_out)
            return self.policy_nn(fc_out), self.state_value_nn(fc_out)
        else:
            fc_out = self.non_conv_fc_nn(input)
            return self.policy_nn(fc_out), self.state_value_nn(fc_out)

class A2CSingleAgent(Agent):
    def __init__(self, device, writer, lr, gamma, conv,
                 tmax, entropy_weight=0.01, value_weight=1.0, clip_grad_norm=None,
                 save_path=None, load_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.gamma = gamma
        self.conv = conv
        self.tmax = tmax
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.clip_grad_norm = clip_grad_norm
        self.eval = False
        self.save_path = save_path
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        with torch.no_grad():
            probs, _ = self.combined_nn(self.process_state(s)) # (1, action_space_dim)
            probs = probs.cpu().numpy() # (1, action_space_dim)
        probs = np.exp(probs) # (1, action_space_dim)
        actions = np.array([
            np.random.choice(self.action_space_size, p=probs[i]) for i in range(self.num_envs)
        ]) # (1,)
        return actions
    
    def reset_transitions(self):
        self.transitions = {
            "s": [], # (tmax, state_space_dim)
            "a": [], # (tmax,)
            "r": [], # (tmax,)
            "sprime": [], # (tmax, state_space_dim)
            "done": [] # (tmax,)
        }

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs

        self.time_step = 0
        
        self.reset_transitions()
        
        self.reward_history = []
        self.current_episode_rewards = 0.0

        self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size, self.conv).to(self.device)
        self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.combined_nn.load_state_dict(checkpoint["nn"])
            self.combined_optimiser.load_state_dict(checkpoint["optimiser"])

    def make_a2c_update(self):

        # policy_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        # state_value_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        
        s = torch.tensor(self.transitions["s"], dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        a = torch.tensor(self.transitions["a"], dtype=torch.int64).to(self.device) # (tmax,)
        r = torch.tensor(self.transitions["r"], dtype=torch.float32).to(self.device) # (tmax,)
        sprime = torch.tensor(self.transitions["sprime"], dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        done = torch.tensor(self.transitions["done"], dtype=torch.float32).to(self.device) # (tmax,)

        is_terminal = done[-1]
        _, last_state_value = self.combined_nn(self.process_state(sprime[-1]).unsqueeze(0))
        last_state_value = last_state_value.squeeze(0)
        R = last_state_value * (1 - is_terminal)

        returns = torch.zeros_like(r).to(self.device) # (tmax,)
        chosen_log_probs = torch.zeros_like(r).to(self.device) # (tmax,)
        advantages = torch.zeros_like(r).to(self.device) # (tmax,)
        state_values = torch.zeros_like(r).to(self.device) # (tmax,)

        for t in reversed(range(self.tmax)):
            R = r[t] * self.gamma + R * (1 - done[t])
            returns[t] = R

            log_probs, state_value = self.combined_nn(self.process_state(s[t]).unsqueeze(0)) # (1, action_space_dim), (1, 1)
            state_value = state_value.squeeze(0)
            chosen_log_prob = log_probs[0, a[t]]
            advantage = R - state_value

            state_values[t] = state_value
            chosen_log_probs[t] = chosen_log_prob
            advantages[t] = advantage
        
        entropy_bonus = self.entropy_weight * (-torch.exp(log_probs) * log_probs).sum(dim=1).mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(chosen_log_probs * advantages.detach()).mean() - entropy_bonus
        state_value_loss = self.value_weight * F.mse_loss(state_values, returns)

        combined_loss = policy_loss + state_value_loss

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()
        
        self.reset_transitions()


    def update(self, s, sprime, a, r, done):

        # s = (1, state_space_dim)
        # sprime = (1, state_space_dim)
        # a = (1,)
        # r = (1,)
        # done = (1,)

        self.current_episode_rewards += r
        self.time_step += 1

        self.transitions["s"].append(s[0])
        self.transitions["a"].append(a[0])
        self.transitions["r"].append(r[0])
        self.transitions["sprime"].append(sprime[0])
        self.transitions["done"].append(done[0])
        
        if self.time_step % self.tmax == 0:
            self.make_a2c_update()

        for env_idx in range(self.num_envs):
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
        return [EnvType.SINGULAR]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]