import os, warnings

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class StateValueNN(nn.Module):
    def __init__(self, state_space_dim):
        super(StateValueNN, self).__init__()
        self.fc1 = nn.Linear(*state_space_dim, 128)
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
        self.fc1 = nn.Linear(*state_space_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = F.log_softmax(self.fc3(f2), dim=1)
        return output

class ReinforceAgent(Agent):
    def __init__(self, device, writer, use_baseline, 
                 policy_lr, state_value_lr, gamma,
                 save_path=None, load_path=None):
        self.device = device
        self.writer = writer
        self.use_baseline = use_baseline
        self.policy_lr = policy_lr
        self.state_value_lr = state_value_lr
        self.eval = False
        self.gamma = gamma
        self.save_path = save_path
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim,)
        with torch.no_grad():
            probs = self.policy_nn.forward(self.process_state(s)).cpu().numpy() # (num_envs, action_space_dim,)
        probs = np.exp(probs) # (num_envs, action_space_dim,)
        return np.array([np.random.choice(self.action_space_size, p=probs[i]) for i in range(self.num_envs)]) # (num_envs,)

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions

        self.transitions = [[] for _ in range(num_envs)]
        self.num_envs = num_envs
        self.current_episode_rewards = np.zeros((self.num_envs,))
        self.reward_history = []
        self.time_step = 0

        self.state_value_nn = StateValueNN(self.state_space_size).to(self.device)
        self.policy_nn = PolicyNN(self.state_space_size, self.action_space_size).to(self.device)
        self.policy_optimiser = optim.Adam(self.policy_nn.parameters(), lr=self.policy_lr)
        self.state_value_optimiser = optim.Adam(self.state_value_nn.parameters(), lr=self.state_value_lr)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False)
            self.policy_nn.load_state_dict(checkpoint["policy_nn"])
            self.state_value_nn.load_state_dict(checkpoint["state_value_nn"])
            self.policy_optimiser.load_state_dict(checkpoint["policy_optimiser"])
            self.state_value_optimiser.load_state_dict(checkpoint["state_value_optimiser"])
    
    def update(self, s, sprime, a, r, done):
        # s is (num_envs, state_space_dim)
        # sprime is (num_envs, state_space_dim)
        # a is (num_envs,)
        # r is (num_envs,)
        # done is (num_envs,)

        self.current_episode_rewards += r # (num_envs,) + (num_envs,)
        for env_idx in range(self.num_envs):
            self.transitions[env_idx].append((s[env_idx], sprime[env_idx], a[env_idx], r[env_idx]))

            if done[env_idx]:
                self.reinforce_update(env_idx)
                self.transitions[env_idx] = []
                self.reward_history.append(self.current_episode_rewards[env_idx])
                if self.writer is not None:
                    self.writer.add_scalar("mean_episode_reward", np.mean(self.reward_history[-100:]), len(self.reward_history))
                    self.writer.add_scalar("episode_reward", self.current_episode_rewards[env_idx], len(self.reward_history))
                self.current_episode_rewards[env_idx] = 0
            
                if self.save_path is not None:
                    torch.save({
                        "policy_nn": self.policy_nn.state_dict(),
                        "state_value_nn": self.state_value_nn.state_dict(),
                        "policy_optimiser": self.policy_optimiser.state_dict(),
                        "state_value_optimiser": self.state_value_optimiser.state_dict(),
                    }, self.save_path)

        self.time_step += 1

    def reinforce_update(self, env_idx):
        # performs a monte carlo update

        env_transitions = self.transitions[env_idx]

        G = 0
        state_value_predictions = torch.tensor([0.0], dtype=torch.float32).to(self.device) 
        policy_loss_total = torch.tensor([0.0], dtype=torch.float32).to(self.device) 

        # accumulate losses
        for t in range(len(env_transitions)-1, -1, -1):
            s, sprime, a, r = env_transitions[t]
            G = self.gamma * G + r

            if self.use_baseline:
                state_value_prediction = self.state_value_nn(self.process_state(s).unsqueeze(0))
                state_value_predictions = state_value_predictions + state_value_prediction
                
            baseline = state_value_prediction if self.use_baseline else 0
            policy_loss_total = policy_loss_total + -(self.gamma**t) * (G - baseline) * self.policy_nn(self.process_state(s).unsqueeze(0)).squeeze(0)[a]

        # backprop policy NN
        self.policy_optimiser.zero_grad()
        self.writer.add_scalar("policy_loss", policy_loss_total.item(), len(self.reward_history))
        policy_loss_total.backward(retain_graph=True)
        self.policy_optimiser.step()

        # backprop state value NN (if used)
        if self.use_baseline:
            self.state_value_optimiser.zero_grad()
            targets = torch.tensor([G], dtype=torch.float32).to(self.device)
            state_value_loss = F.mse_loss(state_value_predictions, targets)
            self.writer.add_scalar("sv_loss", state_value_loss.item(), len(self.reward_history))
            state_value_loss.backward()
            self.state_value_optimiser.step()

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
