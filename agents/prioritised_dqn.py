import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class ReplayMemory:
    def __init__(self, size, alpha, beta, epsilon):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.sum_tree = utils.SumTree(size)
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

class QNet(torch.nn.Module):
    def __init__(self, input_size, output_size, conv):
        super(QNet, self).__init__()

        if conv:
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(3136, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, output_size)
            )
        else:
            self.body = torch.nn.Sequential(
                torch.nn.Linear(*input_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
            )
    
    def forward(self, inp):
        return self.body(inp)

class PrioritisedDQN(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, gamma, epsilon_start, epsilon_end, epsilon_steps, cgn, warmup_steps, alpha, beta):
        self.lr = lr
        self.replay_size = replay_size
        self.C = C
        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.cgn = cgn
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device("cpu")

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())

        self.replay = ReplayMemory(self.replay_size, self.alpha, self.beta, 1e-5)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)

        self._update_target_net()

        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.epsilon_scheduler = utils.LinearScheduler(self.epsilon_start, self.epsilon_end, self.epsilon_steps)

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            if np.random.random() >= self.epsilon:
                q_values = self.qnet(states)
                actions = q_values.argmax(dim=-1)
                return actions
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64).to(self.device)
        
    def _improve(self):
        if self.replay.current_size < self.minibatch_size: return

        minibatch_transitions, sampled_nodes, importance_sampling_weights = self.replay.sample(self.minibatch_size) # (minibatch_size,)
        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch_transitions)
        
        all_s = torch.cat(all_s).to(self.device)
        all_a = torch.cat(all_a).to(self.device)
        all_r = torch.cat(all_r).to(self.device)
        all_sprime = torch.cat(all_sprime).to(self.device)
        all_done = torch.cat(all_done).to(self.device)
        importance_sampling_weights = torch.tensor(np.array(importance_sampling_weights)).to(self.device)

        q_vals = self.qnet(all_s) # (minibatch_size, action_space_dim,)
        chosen_q_vals = q_vals.gather(1, all_a.unsqueeze(1)).squeeze(1) # (minibatch_size,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            greedy_action_selection = self.qnet(all_sprime).argmax(dim=-1) # (minibatch_size,)
            greedy_action_evaluation = self.target_qnet(all_sprime).gather(-1, greedy_action_selection.unsqueeze(1)).squeeze(1)
            targets = all_r + self.gamma * greedy_action_evaluation * (1 - all_done)

        td_error = targets - chosen_q_vals # (minibatch_size)
        self.replay.update_priorities(sampled_nodes, td_error.detach().cpu().numpy())

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(chosen_q_vals, targets, reduction="none") # scalar
        loss = (importance_sampling_weights * loss).mean()
        loss.backward()
        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cgn)
        self.optim.step()

        self.logger.gradient_step_complete(["qnet_loss"], [loss.item()])
        self.logger.network_update({"qnet":self.qnet.state_dict(), "target_qnet":self.target_qnet.state_dict(), "optim":self.optim.state_dict()})


    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None):
        assert env.num_envs == 1
        assert utils.is_space_discrete(env.get_single_action_space())

        utils.seed(seed)
        self.logger = logger
        total_iterations = total_timesteps // self.update_freq
        current_game_states = torch.from_numpy(env.start_states).float().to(self.device)

        self._setup(env)

        for iteration in range(1, total_iterations + 1):

            for current_t in range(self.update_freq):
                self.logger.timestep_complete()

                if self.logger.timesteps_completed % self.C == 0:
                    self._update_target_net()
                self.epsilon = self.epsilon_scheduler.step()
            
                current_actions = self._get_actions(current_game_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)
                        
                current_rewards = torch.from_numpy(current_rewards).float().to(self.device)
                current_sprimes = torch.from_numpy(current_sprimes).float().to(self.device)
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float().to(self.device)

                self.replay.add((
                    current_game_states.detach(),
                    current_actions.detach(),
                    current_rewards,
                    current_sprimes,
                    current_dones,
                ))

                current_game_states = current_sprimes
    
                if self.logger.timesteps_completed > self.warmup_steps:
                    self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        