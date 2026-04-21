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

class A2CTestAgent(Agent):
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
            probs, _ = self.combined_nn(self.process_state(s)) # (num_envs, action_space_dim)
            probs = probs.cpu().numpy() # (num_envs, action_space_dim)
        probs = np.exp(probs) # (num_envs, action_space_dim)
        actions = np.array([
            np.random.choice(self.action_space_size, p=probs[i]) for i in range(self.num_envs)
        ]) # (num_envs,)
        return actions

    def reset_transitions(self):
        self.transitions = {
            "s": [],
            "a": [],
            "r": [],
            "sprime": [],
            "done": []
        }

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs

        self.time_step = 0
        
        self.reset_transitions()
        
        self.reward_history = []
        self.current_episode_rewards = np.zeros((self.num_envs,))

        self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size, self.conv).to(self.device)
        self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.combined_nn.load_state_dict(checkpoint["nn"])
            self.combined_optimiser.load_state_dict(checkpoint["optimiser"])

    def make_a2c_update(self):
        policy_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        state_value_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        entropy_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar

        all_s = torch.as_tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        all_a = torch.as_tensor(np.array(self.transitions["a"]), dtype=torch.int64).to(self.device) # (tmax, num_envs)
        all_r = torch.as_tensor(np.array(self.transitions["r"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        all_sprime = torch.as_tensor(np.array(self.transitions["sprime"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        all_done = torch.as_tensor(np.array(self.transitions["done"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)

        # initialise R by bootstrapping if terminal
        _, last_state_values = self.combined_nn(all_sprime[-1]) # (num_envs, 1)
        last_state_values = last_state_values.view(-1) # (num_envs,)
        R = last_state_values * (1.0 - all_done[-1]) # (num_envs,)

        returns = torch.zeros_like(all_r).to(self.device) # (tmax, num_envs)

        for t in reversed(range(self.tmax)):
            t_s = all_s[t] # (num_envs, state_space_dim)
            t_a = all_a[t] # (num_envs,)
            t_r = all_r[t] # (num_envs,)
            t_done = all_done[t] # (num_envs,)

            # update R per env
            R = t_r + self.gamma * R * (1.0 - t_done) # (num_envs,)
            returns[t] = R


        policy_log_probs, state_values = self.combined_nn(all_s) # (tmax, num_envs, action_space_dim), (tmax, num_envs, 1)
        state_values = state_values.view(self.tmax, self.num_envs) # (tmax, num_envs)

        advatages = returns - state_values # (tmax, num_envs)
        log_probs = policy_log_probs[torch.arange(self.tmax).unsqueeze(1), torch.arange(self.num_envs).unsqueeze(0), all_a] # (tmax, num_envs)

        policy_loss_total = -(log_probs * advatages.detach()).mean() # scalar
        state_value_loss_total = F.mse_loss(state_values, returns) # scalar
        entropy_total = -(torch.exp(policy_log_probs) * policy_log_probs).sum(dim=2).mean() # scalar
        
        combined_loss = policy_loss_total.mean() + (self.value_weight * state_value_loss_total.mean()) - (self.entropy_weight * entropy_total.mean())

        if self.writer != None:
            self.writer.add_scalar("policy_loss", policy_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", state_value_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("policy_entropy", entropy_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("combined_loss", combined_loss.item(), len(self.reward_history) + self.time_step)

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()

        self.reset_transitions()


    def update(self, s, sprime, a, r, done):

        # s = (num_envs, state_space_dim)
        # sprime = (num_envs, state_space_dim)
        # a = (num_envs,)
        # r = (num_envs,)
        # done = (num_envs,)

        self.current_episode_rewards += r
        self.time_step += 1

        self.transitions["s"].append(s)
        self.transitions["a"].append(a)
        self.transitions["r"].append(r)
        self.transitions["sprime"].append(sprime)
        self.transitions["done"].append(done)
        
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
        return [EnvType.SINGULAR, EnvType.VECTORISED]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]