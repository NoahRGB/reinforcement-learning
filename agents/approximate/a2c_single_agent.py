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
    def __init__(self, device, writer, lr, gamma, lam, conv,
                 tmax, entropy_weight, value_weight=1.0, decay_steps=None, decay_rate=0.99, 
                 clip_grad_norm=None, save_path=None, load_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.lam = lam
        self.gamma = gamma
        self.conv = conv
        self.tmax = tmax
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.clip_grad_norm = clip_grad_norm
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
        # self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)
        self.combined_optimiser = optim.RMSprop(self.combined_nn.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.combined_optimiser, step_size=self.decay_steps, gamma=self.decay_rate)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.combined_nn.load_state_dict(checkpoint["nn"])
            self.combined_optimiser.load_state_dict(checkpoint["optimiser"])

    def make_a2c_update(self):
        
        s = torch.tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        a = torch.tensor(np.array(self.transitions["a"]), dtype=torch.int64).to(self.device) # (tmax,)
        r = torch.tensor(np.array(self.transitions["r"]), dtype=torch.float32).to(self.device) # (tmax,)
        sprime = torch.tensor(np.array(self.transitions["sprime"]), dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        done = torch.tensor(np.array(self.transitions["done"]), dtype=torch.float32).to(self.device) # (tmax,)

        log_probs, state_values = self.combined_nn(s) # (tmax, 1)
        state_values = state_values.squeeze(1) # (tmax,)
        chosen_log_probs = log_probs[range(self.tmax), a] # (tmax,)

        returns = torch.zeros_like(r).to(self.device) # (tmax,)
        is_terminal = done[-1]
        _, last_state_value = self.combined_nn(sprime[-1].unsqueeze(0))
        last_state_value = last_state_value.squeeze(0)
        R = last_state_value * (1 - is_terminal)
        for t in reversed(range(self.tmax)):
            R = r[t] * self.gamma + R * (1 - done[t])
            returns[t] = R

        advantages = torch.zeros_like(r).to(self.device)
        for t in range(self.tmax):
            adv = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            for l in range(0, self.tmax-t-1):
                delta = r[t+l] + self.gamma * state_values[t+l+1] - state_values[t+l]
                adv += (self.gamma * self.lam)**l * delta
            advantages[t] = adv

        # advantages = returns - state_values # (tmax,)

        entropy_sum = (-torch.exp(chosen_log_probs) * chosen_log_probs).sum() # (tmax,)
        entropy_bonus = self.entropy_weight * entropy_sum

        policy_loss = -(chosen_log_probs * advantages.detach()).mean()
        state_value_loss = self.value_weight * F.mse_loss(state_values, returns)

        combined_loss = policy_loss - entropy_bonus + state_value_loss

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()

        if self.decay_steps is not None:
            # self.scheduler.step()
            if self.time_step % self.decay_steps == 0:
                self.entropy_weight *= self.decay_rate


        if self.writer is not None:
            self.writer.add_scalar("policy_loss", policy_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", state_value_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("combined_loss", combined_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("entropy", entropy_sum.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0], len(self.reward_history) + self.time_step)
        
        self.reset_transitions()

    def update(self, s, sprime, a, r, done):

        # s = (1, state_space_dim)
        # sprime = (1, state_space_dim)
        # a = (1,)
        # r = (1,)
        # done = (1,)

        self.current_episode_rewards += r
        self.time_step += 1
        
        if self.writer is not None:
            self.writer.add_scalar("entropy_weight", self.entropy_weight, len(self.reward_history) + self.time_step)

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

    def get_supported_env_types(self):
        return [EnvType.SINGULAR]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]