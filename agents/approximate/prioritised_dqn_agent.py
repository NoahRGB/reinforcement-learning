from collections import deque
import random

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType
from utils.sum_tree import SumTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class ReplayMemory:
    def __init__(self, size, alpha, beta, epsilon):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.sum_tree = SumTree(size)
        self.max_observed_priority = 1.0

        self.transitions = np.zeros((self.size), dtype=object)

        self.current_size = 0
        self.next_data = 0
    
    def add(self, transition):
        self.transitions[self.next_data] = transition

        self.sum_tree.add(self.max_observed_priority)

        if self.current_size < self.size:
            self.current_size += 1

        self.next_data = (self.next_data + 1) % self.size
    
    def sample(self, batch_size):
        total_priority = self.sum_tree.get_total_priority()
        sampled_leaves = []
        sampled_nodes = []
        sampled_priorities = []

        for sample in range(batch_size):
            random_priority_value = np.random.uniform(0, total_priority)
            priority, leaf_num, node_idx = self.sum_tree.get_leaf(random_priority_value)
            sampled_leaves.append(leaf_num)
            sampled_nodes.append(node_idx)
            sampled_priorities.append(priority)

        probabilities = np.array(sampled_priorities) / self.sum_tree.get_total_priority()
        importance_sampling_weights = (self.current_size * probabilities) ** (-self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        sampled_transitions = self.transitions[sampled_leaves]

        return sampled_transitions, sampled_nodes, importance_sampling_weights

    def update_priorities(self, sampled_nodes, td_errors):
        for i in range(len(sampled_nodes)):
            new_priority = (abs(td_errors[i]) + self.epsilon) ** self.alpha
            self.max_observed_priority = max(self.max_observed_priority, new_priority)
            self.sum_tree.update_node(sampled_nodes[i], new_priority)

class DQN(nn.Module):
    def __init__(self, conv, state_space_dim, action_space_dim):
        super(DQN, self).__init__()
        self.conv = conv

        self.conv_dqn = nn.Sequential(
            nn.Conv2d(state_space_dim[0], 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(), # (3136,)
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_dim),
        )
        
        if not conv:
            self.regular_dqn = nn.Sequential(
                nn.Linear(*state_space_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_space_dim),
            )

    def forward(self, input):
        if self.conv:
            norm_input = input / 255.0
            return self.conv_dqn(norm_input)
        else:
            return self.regular_dqn(input)

class PrioritisedDQNAgent(Agent):
    def __init__(self, device, logger, lr, conv, replay_memory_size, replay_warmup_length,
                 C, minibatch_size, gamma, alpha, beta,
                 epsilon_start, epsilon_end, epsilon_decay_steps, 
                 clip_grad_norm, update_freq, 
                 load_nn_path=None, save_nn=False, job_title="dqn"):
        
        self.device = device
        self.logger = logger
        self.job_title = job_title
        self.lr = lr
        self.conv = conv
        self.replay_memory_size = replay_memory_size
        self.replay_warmup_length = replay_warmup_length 
        self.minibatch_size = minibatch_size
        self.clip_grad_norm = clip_grad_norm
        self.C = C
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.update_freq = update_freq

        self.epsilon = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end 
        self.epsilon_decay_steps = epsilon_decay_steps 

        self.load_nn_path = load_nn_path
        self.save_nn = save_nn

    def process_state(self, s):
        return torch.tensor(s, dtype=torch.float32).to(self.device)

    def clone_qnet(self):
        # copy DQN to target DQN
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.start_state = start_state
        self.num_envs = num_envs
        self.state_space_dim = state_space.dimensions 
        self.action_space_dim = action_space.dimensions 

        self.actions = [i for i in range(self.action_space_dim)] # discrete actions!
        self.current_episode_rewards = np.zeros((self.num_envs,))
        self.time_step = 0
        self.num_updates = 0
        self.reward_history = []

        # create DQN, target DQN, optimiser, replay memory
        self.dqn = DQN(self.conv, self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_dqn = DQN(self.conv, self.state_space_dim, self.action_space_dim).to(self.device)
        self.clone_qnet()
        self.optimiser = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.replay = ReplayMemory(self.replay_memory_size, self.alpha, self.beta, 1e-5)

        # load relevant models if necessary
        if self.load_nn_path is not None:
            checkpoint = torch.load(self.load_nn_path, weights_only=False, map_location=self.device)
            self.dqn.load_state_dict(checkpoint["dqn"])
            self.target_dqn.load_state_dict(checkpoint["target_dqn"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.time_step = checkpoint["time_step"]
            self.epsilon = checkpoint["epsilon"]

    def episode_terminated(self, env_idx):
        self.reward_history.append(self.current_episode_rewards[env_idx])
        self.logger.log("mean_episode_reward", np.mean(self.reward_history[-100:]), len(self.reward_history))
        self.logger.log("episode_reward", self.current_episode_rewards[env_idx], len(self.reward_history))
        self.current_episode_rewards[env_idx] = 0.0

        if self.save_nn:
            self.logger.save_torch({
                "dqn": self.dqn.state_dict(),
                "target_dqn": self.target_dqn.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "time_step": self.time_step,
                "epsilon": self.epsilon,
            }, f"{self.job_title}_model")
        self.logger.save_logs()

    def finish_episode(self, episode_num):
        # self.episode_terminated(0)
        pass

    def get_best_actions(self, s):
        # s is (num_envs, state_space_dim,)
        # forward s through DQN and return the action with the highest q val for every env
        with torch.no_grad():
            qvals = self.dqn(self.process_state(s)).cpu().numpy() # (num_envs, action_space_dim,)
        return qvals.argmax(axis=1) # (num_envs,)

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim,)
        return self.generate_action(s)

    def generate_action(self, s):
        # s is (num_envs, state_space_dim,)
        # epsilon greedy action selection
        if np.random.random() >= self.epsilon:
            return self.get_best_actions(s) # (num_envs,)
        return [np.random.choice(self.actions) for _ in range(self.num_envs)] # (num_envs,)
    
    def replay_memory_update(self):
        # performs an update on a sample of the transitions stored in replay memory

        minibatch_transitions, sampled_nodes, importance_sampling_weights = self.replay.sample(self.minibatch_size) # (minibatch_size,)

        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch_transitions)
        all_s = torch.stack([self.process_state(s_) for s_ in all_s]).to(self.device) # (minibatch_size, state_space_dim,)
        all_a = torch.tensor(all_a, dtype=torch.int64).to(self.device) # (minibatch_size,)
        all_r = torch.tensor(all_r, dtype=torch.float32).to(self.device) # (minibatch_size,)
        all_sprime = torch.stack([self.process_state(s_) for s_ in all_sprime]).to(self.device) # (minibatch_size, state_space_dim,)
        all_done = torch.tensor(all_done, dtype=torch.float32).to(self.device) # (minibatch_size,)
        importance_sampling_weights = torch.tensor(importance_sampling_weights).to(self.device)

        q_vals = self.dqn(all_s) # (minibatch_size, action_space_dim,)
        chosen_q_vals = q_vals.gather(1, all_a.unsqueeze(1)).squeeze(1) # (minibatch_size,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            greedy_action_selection = self.dqn(all_sprime).argmax(dim=-1) # (minibatch_size,)
            greedy_action_evaluation = self.target_dqn(all_sprime).gather(-1, greedy_action_selection.unsqueeze(1)).squeeze(1)
            targets = all_r + self.gamma * greedy_action_evaluation * (1 - all_done)

        td_error = targets - chosen_q_vals # (minibatch_size,)
        self.replay.update_priorities(sampled_nodes, td_error.detach().cpu().numpy())

        # zero grads, calculate loss, backprop, optimiser step
        self.optimiser.zero_grad()
        loss = F.smooth_l1_loss(chosen_q_vals, targets, reduction="none") # minibatch_size
        loss = (importance_sampling_weights * loss).mean()
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.clip_grad_norm)
        self.optimiser.step()

        # log losses/qvals
        self.logger.log("loss", loss.item(), self.num_updates)
        self.logger.log("avg_qval", chosen_q_vals.mean().item(), self.num_updates)
        self.logger.save_logs()

        self.num_updates += 1

    def update(self, s, sprime, a, r, done):
        # s = (num_envs, state_space_dim,)
        # sprime = (num_envs, state_space_dim,)
        # a = (num_envs,)
        # r = (num_envs,)
        # done = (num_envs,)

        self.current_episode_rewards += r # (num_envs,) + (num_envs,)

        self.time_step += 1
        
        for env_idx in range(self.num_envs):
            # add the transition to replay memory for every env
            self.replay.add((s[env_idx], a[env_idx], r[env_idx], sprime[env_idx], done[env_idx]))

            # if the env episode ended, save/log/reset rewards
            if done[env_idx]:
                self.episode_terminated(env_idx)

        # if replay memory has enough samples
        if self.replay.current_size >= self.replay_warmup_length and self.replay.current_size >= self.minibatch_size:
            
            # decay epsilon
            self.epsilon = (self.epsilon_end + (self.epsilon_start - self.epsilon_end) *
                            (1 - ((self.time_step - self.replay_warmup_length) / self.epsilon_decay_steps)))
            self.epsilon = max(self.epsilon, self.epsilon_end)
            self.logger.log("epsilon", self.epsilon, self.time_step)

            # every update_freq time steps perform an update from replay memory
            if self.time_step % self.update_freq == 0:
                self.replay_memory_update()

        # clone the dqn every C time steps
        if self.time_step % self.C == 0:
            self.clone_qnet()

    def get_supported_env_types(self):
        return [EnvType.SINGULAR, EnvType.VECTORISED]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
    
    def get_dump(self):
        return f"""
        title: {self.job_title}
        epsilon: {self.epsilon}
        time_step: {self.time_step}
        replay_memory_size: {self.replay.current_size}/{self.replay_memory_size}
        conv: {self.conv}
        lr: {self.lr}
        replay_warmup_length: {self.replay_warmup_length}
        C: {self.C}
        minibatch_size: {self.minibatch_size}
        gamma: {self.gamma}
        epsilon_start: {self.epsilon_start}
        epsilon_end: {self.epsilon_end}
        epsilon_decay_steps: {self.epsilon_decay_steps}
        clip_grad_norm: {self.clip_grad_norm}
        update_freq: {self.update_freq}
        load_nn_path: {self.load_nn_path}
        save_nn: {self.save_nn}
        """