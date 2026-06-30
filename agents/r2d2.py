import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.nstep_buffer = deque(maxlen=self.n)

    def append(self, s, a, r, sprime, done, hidden_state):
        self.nstep_buffer.append((s, a, r, sprime, done, hidden_state))

    def is_full(self):
        return len(self.nstep_buffer) == self.n
    
    def clear(self):
        self.nstep_buffer.clear()

    def nsteps_done(self):
        all_s, all_a, all_r, all_sprime, all_done, all_hidden = zip(*self.nstep_buffer)
        all_s = torch.stack(all_s).squeeze(1) # (n, state_dim,)
        all_a = torch.stack(all_a).squeeze(1) # (n,)
        all_r = torch.stack(all_r).squeeze(1) # (n,)
        all_sprime = torch.stack(all_sprime).squeeze(1) # (n, state_dim,)
        all_done = torch.stack(all_done).squeeze(1) # (n,)
        initial_hidden_state = (
            all_hidden[0][0],
            all_hidden[0][1],
        )

        final_timestep = (len(all_r) - 1) if self.n > 1 else 0
        for t in range(len(all_r)):
            if all_done[t]:
                final_timestep = t
                break

        G = 0
        for t in reversed(range(final_timestep + 1)):
            G = all_r[t] + self.gamma * G

        return (all_s[0], all_a[0], G, all_sprime[final_timestep], all_done[final_timestep], initial_hidden_state)
        

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

    def __len__(self):
        return self.current_size
    
    def append(self, transition):
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
    def __init__(self, input_size, output_size, conv, use_dueling, lstm_size):
        super(QNet, self).__init__()
        self.conv = conv
        self.use_dueling = use_dueling
        self.lstm_size = lstm_size

        if conv:
            self.conv_body = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(), # 3136
            )

            self.lstm = torch.nn.LSTM(3136, self.lstm_size, batch_first=True)
        else:
            self.lstm = torch.nn.LSTM(*input_size, self.lstm_size, batch_first=True)

        self.fc = torch.nn.Linear(self.lstm_size, output_size)
        self.state_value_head = torch.nn.Linear(self.lstm_size, 1)
        self.advantage_head = torch.nn.Linear(self.lstm_size, output_size)
    
    def forward(self, inp, inp_hidden=None):
        # input is (batch_size (B), sequence_length (T), state_space_dim)
        # e.g. (32, T, 1, 84, 84)

        if self.conv:
            # input is (batch_size (B), sequence_length (T), channels (C), height (H), width (W))
            batch_size, seq_len, channels, height, width = inp.shape
            norm_input = inp / 255.0

            # merge batch/time for conv layer
            conv_out = self.conv_body(norm_input.view(batch_size * seq_len, channels, height, width))

            # restore batch/time for lstm layer (with conv output)
            lstm_out, hidden = self.lstm(conv_out.view(batch_size, seq_len, 3136), inp_hidden)
        else:
            lstm_out, hidden = self.lstm(inp, inp_hidden)

        if self.use_dueling:
            state_value = self.state_value_head(lstm_out)
            advantage = self.advantage_head(lstm_out)
            qvals_out = state_value + (advantage - torch.mean(advantage, axis=-1, keepdim=True))
        else:
            qvals_out = self.fc(lstm_out)

        return qvals_out, hidden

class R2D2(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, 
                 gamma, epsilon_scheduler: utils.LinearScheduler, cgn, 
                 warmup_steps, gradient_steps, seq_len, overlap, eta, alpha,
                 beta_scheduler: utils.LinearScheduler, nsteps,
                 lstm_size=64,  use_dueling=False, use_double=False, 
                 use_per=False, load_path=None):
        
        self.lr = lr
        self.replay_size = replay_size
        self.C = C
        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = epsilon_scheduler.get_value()
        self.cgn = cgn
        self.seq_len = seq_len
        self.overlap = overlap
        self.burn_in_steps = seq_len // 2
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.device = torch.device("cpu")
        self.eta = eta
        self.alpha = alpha
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.use_per = use_per
        self.beta_scheduler = beta_scheduler
        self.nsteps = nsteps
        self.lstm_size = lstm_size
        self.beta = beta_scheduler.get_value()
        self.load_path = load_path

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.sequence_buffer = deque(maxlen=self.seq_len)
        self.nstep_buffer = NStepBuffer(self.nsteps, self.gamma)

        if self.use_per:
            self.replay = ReplayMemory(self.replay_size//self.seq_len, self.alpha, self.beta, 1e-5)
        else:
            self.replay = deque(maxlen=self.replay_size//self.seq_len)

        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv, self.use_dueling, self.lstm_size).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv, self.use_dueling, self.lstm_size).to(self.device)

        self._update_target_net()

        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        # self.optim = torch.optim.RMSprop(self.qnet.parameters(), lr=self.lr, momentum=0.95)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False, map_location=self.device)
            self.qnet.load_state_dict(checkpoint["qnet"])
            self.target_qnet.load_state_dict(checkpoint["target_qnet"])
            self.optim.load_state_dict(checkpoint["optim"])

    def _get_actions(self, states: torch.Tensor, running_hidden_states: tuple):
        with torch.no_grad():
            states_input = states.unsqueeze(1) # (num_envs, 1, state_dim) add fake time/seq dim
            hidden_input = (
                running_hidden_states[0].to(self.device),
                running_hidden_states[1].to(self.device)
            )

            q_values, new_running_hidden_states = self.qnet(states_input, hidden_input)
            if np.random.random() >= self.epsilon:
                q_values = q_values.squeeze(1) # (num_envs, action_dim) remove fake time/seq dim
                actions = q_values.argmax(dim=-1)
                return actions, new_running_hidden_states
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64), new_running_hidden_states
        
    def _improve(self):
        if len(self.replay) < self.minibatch_size: return

        if self.use_per:
            minibatch, sampled_nodes, importance_sampling_weights = self.replay.sample(self.minibatch_size) # (minibatch_size,)
        else:
            minibatch = random.sample(self.replay, self.minibatch_size)

        burnin_start_index = 0
        burnin_end_index = self.burn_in_steps
        learnable_start_index = self.burn_in_steps
        learnable_end_index = self.seq_len

        all_s = torch.stack([torch.stack([t[0] for t in seq[0]]).view(self.seq_len, *self.state_space_dim) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len, state_space_dim)
        all_a = torch.stack([torch.stack([t[1] for t in seq[0]]).view(self.seq_len) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len)
        all_r = torch.stack([torch.stack([t[2] for t in seq[0]]).view(self.seq_len) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len)
        all_sprime = torch.stack([torch.stack([t[3] for t in seq[0]]).view(self.seq_len, *self.state_space_dim) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len, state_space_dim)
        all_done = torch.stack([torch.stack([t[4] for t in seq[0]]).view(self.seq_len) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len)
        all_initial_hidden_states = (
            torch.cat([seq[1][0] for seq in minibatch], dim=1).to(self.device), # (minibatch_size, 1, 512)
            torch.cat([seq[1][1] for seq in minibatch], dim=1).to(self.device) # (minibatch_size, 1, 512)
        )

        learnable_s = all_s[:, learnable_start_index:learnable_end_index, :] # (minibatch_size, learnable_steps, state_space_dim)
        learnable_a = all_a[:, learnable_start_index:learnable_end_index] # (minibatch_size, learnable_steps)
        learnable_r = all_r[:, learnable_start_index:learnable_end_index] # (minibatch_size, learnable_steps)
        learnable_sprime = all_sprime[:, learnable_start_index:learnable_end_index, :] # (minibatch_size, learnable_steps, state_space_dim)
        learnable_done = all_done[:, learnable_start_index:learnable_end_index] # (minibatch_size, learnable_steps)
        masks = 1 - learnable_done # (minibatch_size, learnable_steps)

        with torch.no_grad():
            burnin_s = all_s[:, burnin_start_index:burnin_end_index, :] # (minibatch_size, burnin_steps, state_space_dim)
            _, burnin_hidden_state_out = self.qnet(burnin_s, all_initial_hidden_states) # (minibatch_size, burnin_steps, action_space_dim), (minibatch_size, 1, 512)

        q_vals, state_value_hidden_state_out = self.qnet(learnable_s, burnin_hidden_state_out) # (minibatch_size, seq_len, action_space_dim,)
        chosen_q_vals = q_vals.gather(-1, learnable_a.unsqueeze(-1)).squeeze(-1) # (minibatch_size, seq_len)

        # compute the target values (using the target DQN)
        with torch.no_grad():

            if self.use_double:

                greedy_action_qvals, _ = self.qnet(learnable_sprime, state_value_hidden_state_out) # (minibatch_size,)
                greedy_action_selection = greedy_action_qvals.argmax(dim=-1) # (minibatch_size,)

                greedy_action_evaluation_qvals, _ = self.target_qnet(learnable_sprime, state_value_hidden_state_out)
                greedy_action_evaluation = greedy_action_evaluation_qvals.gather(-1, greedy_action_selection.unsqueeze(-1)).squeeze(-1) # (minibatch_size,)

                targets = learnable_r + (self.gamma ** self.nsteps) * greedy_action_evaluation * masks # (minibatch_size, seq_len,)

            else:

                target_qvals, target_hidden = self.target_qnet(learnable_sprime, state_value_hidden_state_out) # (minibatch_size, unroll_iterations, action_space_dim,)
                targets = learnable_r + (self.gamma ** self.nsteps) * target_qvals.max(-1)[0] * masks # (minibatch_size, unroll_iterations,)

        if self.use_per:
            td_errors = (targets - chosen_q_vals).abs() # (minibatch_size, seq_len)
            max_td_errors = torch.max(td_errors, dim=-1).values # (minibatch_size)
            mean_td_errors = torch.mean(td_errors, dim=-1) # (minibatch_size)
            priorities = self.eta * max_td_errors + (1 - self.eta) * mean_td_errors
            self.replay.update_priorities(sampled_nodes, priorities.detach().cpu().numpy())

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()
        loss = torch.nn.functional.mse_loss(chosen_q_vals, targets, reduction="none") # scalar

        if self.use_per:
            importance_sampling_weights = torch.tensor(np.array(importance_sampling_weights)).to(self.device)
            loss = (importance_sampling_weights * loss.mean(dim=-1)).mean()
        else:
            loss = loss.mean()

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

        running_hidden_states = (
            torch.zeros((1, 1, self.lstm_size)).to(self.device), 
            torch.zeros((1, 1, self.lstm_size)).to(self.device)
        )

        for iteration in range(1, total_iterations + 1):

            for current_t in range(self.update_freq):
                self.logger.timestep_complete()

                if self.logger.timesteps_completed % self.C == 0:
                    self._update_target_net()
                self.epsilon = self.epsilon_scheduler.step()

                if self.use_per:
                    self.beta = self.beta_scheduler.step()
                    self.replay.beta = self.beta
            
                current_actions, new_running_hidden_states = self._get_actions(current_game_states, running_hidden_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())
     
                current_rewards = torch.from_numpy(current_rewards).float()
                current_sprimes = torch.from_numpy(current_sprimes).float().to(self.device)
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float()

                self.nstep_buffer.append(
                    current_game_states.detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards.detach().cpu(),
                    current_sprimes.detach().cpu(),
                    current_dones.detach().cpu(),
                    running_hidden_states
                )

                if self.nstep_buffer.is_full():
                    self.sequence_buffer.append(self.nstep_buffer.nsteps_done())

                running_hidden_states = new_running_hidden_states

                if len(self.sequence_buffer) == self.seq_len:

                    # add the sequence of transitions into replay memory
                    # with the hidden state of the first transition
                    self.replay.append((
                        self.sequence_buffer.copy(),
                        self.sequence_buffer[0][5]
                    ))

                    # remove from the left of the seq buffer so
                    # it can be refilled based on how much overlap
                    # is specified
                    for _ in range(self.overlap):
                        self.sequence_buffer.popleft()


                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)

                        # discard any ongoing sequences, reset hidden states
                        self.sequence_buffer.clear()
                        self.nstep_buffer.clear()
                        running_hidden_states = (
                            torch.zeros((1, 1, self.lstm_size)).to(self.device), 
                            torch.zeros((1, 1, self.lstm_size)).to(self.device)
                        )

                current_game_states = current_sprimes
                
                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve()

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        