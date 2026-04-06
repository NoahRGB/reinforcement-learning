from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CombinedNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(CombinedNN, self).__init__()

        self.fc_nn = nn.Sequential(
            nn.Linear(*state_space_dim, 128),
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
        out = self.fc_nn(input)
        return self.policy_nn(out), self.state_value_nn(out)

class A2CAgent(Agent):
    def __init__(self, device, writer, lr, gamma, tmax, entropy_weight=0.01,
                 save_nn_path=None, load_nn_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.entropy_weight = entropy_weight
        self.tmax = tmax
        self.eval = False
        self.gamma = gamma
        self.save_nn_path = save_nn_path
        self.load_nn_path = load_nn_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        with torch.no_grad():
            probs, _ = self.combined_nn(self.process_state(s)) # (num_envs, 6)
            probs = probs.cpu().numpy() # (num_envs, 6)
        probs = np.exp(probs) # (num_envs, 6)
        actions = np.array([np.random.choice(self.action_space_size, p=probs[i]) for i in range(probs.shape[0])]) # (num_envs,)
        return actions
    
    def reset_transitions(self):
        self.transitions = {
            "s":[],
            "a":[],
            "r":[],
            "sprime":[],
            "done":[]
        }

    def initialise(self, state_space, action_space, start_state, num_envs,resume=False):
        
        self.reset_transitions()

        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.state_space_mins = state_space.min_bounds
        self.state_space_maxs = state_space.max_bounds
        self.num_envs = num_envs

        if not resume:
            self.time_step = 0

            self.total_episodes_completed = 0
            self.reward_history = []
            self.current_episode_rewards = np.zeros((self.num_envs,))

            self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size).to(self.device)
            self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)

            # load saved models
            if self.load_nn_path != None:
                self.combined_nn.load_state_dict(torch.load(self.load_nn_path))

    def make_update(self):
        all_policy_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_state_value_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_entropy_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)

        # self.transitions["s"] = (t_max, num_envs, 4, 84, 84)
        # self.transitions["sprime"] = (t_max, num_envs, 4, 84, 84)
        # self.transitions["a"] = (t_max, num_envs,)
        # self.transitions["r"] = (t_max, num_envs,)
        # self.transitions["done"] = (t_max, num_envs,)

        with torch.no_grad():
            last_terms = self.transitions["done"][-1] # (num_envs,) 
            last_sprimes = self.transitions["sprime"][-1] # (num_envs, 4, 84, 84)
            _, last_state_bootstraps = self.combined_nn(self.process_state(last_sprimes)) # (num_envs, 6), (num_envs, 1)
            R = (last_state_bootstraps.cpu().squeeze(1) * (1 - last_terms)).squeeze().float().to(self.device) # (num_envs,)

        for t in reversed(range(self.tmax)):
            
            all_states_t = self.transitions["s"][t] # (num_envs, 4, 84, 84)
            all_actions_t = self.transitions["a"][t] # (num_envs,)
            all_rewards_t = torch.as_tensor(self.transitions["r"][t], dtype=torch.float32).to(self.device) # (num_envs,)
            all_dones_t = torch.as_tensor(self.transitions["done"][t], dtype=torch.float32).to(self.device) # (num_envs,)

            # (num_envs,) + scalar * (num_envs,) = (num_envs,)
            R = all_rewards_t + self.gamma * R * (1 - all_dones_t) # (num_envs,)
            # R = all_rewards_t + self.gamma * R

            all_policy_predictions, all_state_value_predictions = self.combined_nn(
                torch.tensor(all_states_t, dtype=torch.float32).to(self.device)
            ) # (num_envs, 6), (num_envs, 1)
            
            # (num_envs,) - (num_envs,) = (num_envs,)
            with torch.no_grad():
                all_advantages = R - all_state_value_predictions.squeeze(1) # (num_envs,)

            all_log_probs = all_policy_predictions[torch.arange(self.num_envs), all_actions_t] # (num_envs,)

            all_policy_loss_total += -(all_advantages * all_log_probs).mean() # scalar
            all_state_value_loss_total += F.mse_loss(all_state_value_predictions.squeeze(1), R) # scalar
            all_entropy_loss_total += self.entropy_weight * -(torch.exp(all_policy_predictions) * all_policy_predictions).sum(dim=1).mean() # scalar

        # average losses over tmax steps (reduces scale of loss)
        all_policy_loss_total /= self.tmax
        all_state_value_loss_total /= self.tmax
        all_entropy_loss_total /= self.tmax

        combined_loss = all_policy_loss_total + all_state_value_loss_total + all_entropy_loss_total # scalar (loss)

        if self.writer != None:
            self.writer.add_scalar("policy_loss", all_policy_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", all_state_value_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("entropy_loss", all_entropy_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("combined_loss", combined_loss.item(), len(self.reward_history) + self.time_step)

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        self.combined_optimiser.step()

        self.reset_transitions()

    def update(self, s, sprime, a, r, done):

        # s = (num_envs, 4, 84, 84)
        # sprime = (num_envs, 4, 84, 84)
        # a = (num_envs,)
        # r = (num_envs,)
        # done = (num_envs,)
        self.transitions["s"].append(s)
        self.transitions["a"].append(a)
        self.transitions["r"].append(r)
        self.transitions["sprime"].append(sprime)
        self.transitions["done"].append(done)

        self.current_episode_rewards += r
        for i in range(self.num_envs):
            if done[i]:
                self.total_episodes_completed += 1
                self.reward_history.append(self.current_episode_rewards[i])
                if self.writer != None:
                    self.writer.add_scalar("mean_episode_reward", np.mean(self.reward_history[-100:]), self.total_episodes_completed)
                    self.writer.add_scalar("episode_reward", self.current_episode_rewards[i], self.total_episodes_completed)
                self.current_episode_rewards[i] = 0.0

        # every tmax steps (or if episode ends), make update
        if (self.time_step+1) % self.tmax == 0:
            self.make_update()

        self.time_step += 1

    def finish_episode(self, episode_num):
        # not used for agents compatible with vectorised environments
        pass

    def toggle_eval(self):
        self.eval = not self.eval

    def get_supported_env_types(self):
        return [EnvType.SINGULAR, EnvType.VECTORISED]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]

