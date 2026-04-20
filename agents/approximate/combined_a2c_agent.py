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
            np.random.choice(self.action_space_size, p=probs[i]) for i in range(self.num_envs)
        ]) # (num_envs,)
        return actions

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs

        self.time_step = 0
        self.transitions = {
            "s":[],
            "a":[],
            "r":[],
            "sprime":[],
            "done":[]
        }
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
        all_policy_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_state_value_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_entropy_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar

        # self.transitions["s"] = (t_max, num_envs, state_space_dim,)
        # self.transitions["sprime"] = (t_max, num_envs, state_space_dim,)
        # self.transitions["a"] = (t_max, num_envs,)
        # self.transitions["r"] = (t_max, num_envs,)
        # self.transitions["done"] = (t_max, num_envs,)

        last_dones = torch.as_tensor(self.transitions["done"][-1], dtype=torch.float32).to(self.device) # (num_envs,) 
        last_sprimes = torch.as_tensor(self.transitions["sprime"][-1], dtype=torch.float32).to(self.device) # (num_envs, state_space_dim,)
        _, last_state_bootstraps = self.combined_nn(last_sprimes) # (num_envs, 6), (num_envs, 1)
        R = (last_state_bootstraps.squeeze(1) * (1 - last_dones)).squeeze() # (num_envs,)

        for t in reversed(range(self.tmax)):
            
            all_states_t = torch.as_tensor(self.transitions["s"][t], dtype=torch.float32).to(self.device) # (num_envs, state_space_dim,)
            all_actions_t = torch.as_tensor(self.transitions["a"][t], dtype=torch.int64).to(self.device) # (num_envs,)
            all_rewards_t = torch.as_tensor(self.transitions["r"][t], dtype=torch.float32).to(self.device) # (num_envs,)
            all_dones_t = torch.as_tensor(self.transitions["done"][t], dtype=torch.float32).to(self.device) # (num_envs,)

            # (num_envs,) + scalar * (num_envs,) = (num_envs,)
            R = all_rewards_t + self.gamma * R * (1 - all_dones_t) # (num_envs,)

            all_policy_predictions, all_state_value_predictions = self.combined_nn(all_states_t) # (num_envs, 6), (num_envs, 1)
            
            # (num_envs,) - (num_envs,) = (num_envs,)
            all_advantages = R - all_state_value_predictions.squeeze(1) # (num_envs,)
            # all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8) # normalise advantages

            all_log_probs = all_policy_predictions[torch.arange(self.num_envs), all_actions_t] # (num_envs,)

            all_policy_loss_total += -(all_advantages.detach() * all_log_probs).mean() # scalar
            all_state_value_loss_total += F.mse_loss(all_state_value_predictions.squeeze(1), R) # scalar
            all_entropy_total += -(torch.exp(all_policy_predictions) * all_policy_predictions).sum(dim=1).mean() # scalar
            
        # average losses over tmax steps (reduces scale of loss)
        all_policy_loss_total /= self.tmax
        all_state_value_loss_total /= self.tmax
        
        all_entropy_mean = all_entropy_total / self.tmax
        all_entropy_bonus = self.entropy_weight * all_entropy_mean

        combined_loss = all_policy_loss_total + all_state_value_loss_total - all_entropy_bonus # scalar (loss)

        if self.writer != None:
            self.writer.add_scalar("policy_loss", all_policy_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", all_state_value_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("mean_policy_entropy", all_entropy_mean.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("combined_loss", combined_loss.item(), len(self.reward_history) + self.time_step)

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()

        self.transitions = {
            "s":[],
            "a":[],
            "r":[],
            "sprime":[],
            "done":[]
        }

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