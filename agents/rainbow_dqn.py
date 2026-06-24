import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class NoisyLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NoisyLinear, self).__init__()

        self.weight_mu = torch.nn.Parameter(torch.empty((input_size, output_size), dtype=torch.float32, requires_grad=True))
        self.weight_sigma = torch.nn.Parameter(torch.empty((input_size, output_size), dtype=torch.float32, requires_grad=True))

        self.bias_mu = torch.nn.Parameter(torch.empty((output_size), dtype=torch.float32, requires_grad=True))
        self.bias_sigma = torch.nn.Parameter(torch.empty((output_size), dtype=torch.float32, requires_grad=True))

        # noisy net paper reccommends initialising mu with uniform distribution
        # of [-1/sqrt(input_size), 1/sqrt(input_size)] and sigma with constant 0.5/sqrt(input_size)
        self.weight_mu.data.uniform_(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size))
        self.weight_sigma.data.fill_(0.5 / np.sqrt(input_size))
        self.bias_mu.data.uniform_(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size))
        self.bias_sigma.data.fill_(0.5 / np.sqrt(input_size))

    def forward(self, inp):
        # y = wx + b

        if self.training:
            # fill the epsilon tensors with mean 0 normal samples
            weight_epsilon = torch.randn(self.weight_mu.shape).to(inp.device)
            bias_epsilon = torch.randn(self.bias_mu.shape).to(inp.device)
        else:
            # fill the epsilon tensors with 0
            # (you want greedy behaviour for evaluation)
            weight_epsilon = torch.zeros(self.weight_mu.shape).to(inp.device)
            bias_epsilon = torch.zeros(self.bias_mu.shape).to(inp.device)

        noisy_net_weight = self.weight_mu + self.weight_sigma * weight_epsilon
        noisy_net_bias = self.bias_mu + self.bias_sigma * bias_epsilon

        return inp @ noisy_net_weight + noisy_net_bias

class ReplayMemory:
    def __init__(self, size, alpha, beta, epsilon, obs_shape, is_conv):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.sum_tree = utils.SumTree(size)
        self.max_observed_priority = 1.0

        # self.transitions = np.zeros((self.size), dtype=object)
        self.states = np.zeros((self.size, *obs_shape), dtype=np.uint8 if is_conv else np.float32)
        self.actions = np.zeros((self.size), dtype=np.int64)
        self.rewards = np.zeros((self.size), dtype=np.float32)
        self.sprimes = np.zeros((self.size, *obs_shape), dtype=np.uint8 if is_conv else np.float32)
        self.dones = np.zeros((self.size), dtype=np.float32)

        self.current_size = 0
        self.next_data = 0
    
    def add(self, transition):
        s, a, G, sprime, done = transition
        self.states[self.next_data] = s.numpy()
        self.actions[self.next_data] = a.item()
        self.rewards[self.next_data] = G.item()
        self.sprimes[self.next_data] = sprime.numpy()
        self.dones[self.next_data] = done.item()

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

        s = torch.from_numpy(self.states[sampled_leaves]).float().pin_memory()
        a = torch.from_numpy(self.actions[sampled_leaves]).long().pin_memory()
        G = torch.from_numpy(self.rewards[sampled_leaves]).float().pin_memory()
        sprime = torch.from_numpy(self.sprimes[sampled_leaves]).float().pin_memory()
        done = torch.from_numpy(self.dones[sampled_leaves]).float().pin_memory()

        return (s, a, G, sprime, done), sampled_nodes, importance_sampling_weights

    def update_priorities(self, sampled_nodes, td_errors):
        for i in range(len(sampled_nodes)):
            new_priority = (abs(td_errors[i]) + self.epsilon) ** self.alpha
            self.max_observed_priority = max(self.max_observed_priority, new_priority)
            self.sum_tree.update_node(sampled_nodes[i], new_priority)

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.nstep_buffer = deque(maxlen=n)

    def add(self, s, a, r, sprime, done):
        self.nstep_buffer.append((s, a, r, sprime, done))

    def is_full(self):
        return len(self.nstep_buffer) == self.n

    def nsteps_done(self):
        all_s, all_a, all_r, all_sprime, all_done = zip(*self.nstep_buffer)
        all_s = torch.stack(all_s).squeeze() # (n, state_dim,)
        all_a = torch.stack(all_a).squeeze() # (n,)
        all_r = torch.stack(all_r).squeeze() # (n,)
        all_sprime = torch.stack(all_sprime).squeeze() # (n, state_dim,)
        all_done = torch.stack(all_done).squeeze() # (n,)

        final_timestep = len(all_r) - 1
        for t in range(len(all_r)):
            if all_done[t]:
                final_timestep = t
                break

        G = 0
        for t in reversed(range(final_timestep + 1)):
            G = all_r[t] + self.gamma * G

        return (all_s[0], all_a[0], G, all_sprime[final_timestep], all_done[final_timestep])
        

class QNet(torch.nn.Module):
    def __init__(self, input_size, action_dim, num_atoms, is_conv, is_distributional):
        super(QNet, self).__init__()
        self.conv = is_conv
        self.distributional = is_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        if self.conv:
            self.body_out_size = 512
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                NoisyLinear(3136, 512),
                torch.nn.ReLU(),
            )
        else:
            self.body_out_size = 256
            self.body = torch.nn.Sequential(
                NoisyLinear(*input_size, 256),
                torch.nn.ReLU(),
                NoisyLinear(256, 256),
                torch.nn.ReLU(),
            )
        
        if self.distributional:
            self.state_value_head = NoisyLinear(self.body_out_size, num_atoms)
            self.advantage_head = NoisyLinear(self.body_out_size, action_dim*num_atoms)
        else:
            self.state_value_head = NoisyLinear(self.body_out_size, 1)
            self.advantage_head = NoisyLinear(self.body_out_size, action_dim)      

    def forward(self, inp):
        batch, *state_dims = inp.shape

        if self.conv:
            new_inp = inp / 255.0
            body_out = self.body(new_inp)
        else:   
            body_out = self.body(inp)

        state_value_out = self.state_value_head(body_out) # (batch, num_atoms)
        advantage_out = self.advantage_head(body_out) # (batch, action_dim*num_atoms)

        if self.distributional:
            state_value_out = state_value_out.view(batch, 1, self.num_atoms) # (batch, 1, num_atoms)
            advantage_out = advantage_out.view(batch, self.action_dim, self.num_atoms) # (batch, action_dim, num_atoms)
       
        all_out = state_value_out + (advantage_out - torch.mean(advantage_out, axis=1, keepdim=True))
        return all_out

class RainbowDQN(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, gamma, 
                 cgn, warmup_steps, gradient_steps, use_distributional, vmin, 
                 vmax, N, nstep, alpha, beta_scheduler: utils.LinearScheduler,
                 load_path=None):
        self.lr = lr
        self.replay_size = replay_size
        self.C = C
        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.cgn = cgn
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.use_distributional = use_distributional
        self.device = torch.device("cpu")
        self.load_path = load_path
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = N
        self.nstep = nstep
        self.alpha = alpha
        self.beta_scheduler = beta_scheduler
        self.beta = beta_scheduler.get_value()
        self.delta_z = (self.vmax - self.vmin) / (self.num_atoms - 1)
        self.atoms = np.array([self.vmin + i * self.delta_z for i in range(self.num_atoms)])

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.nstep_buffer = NStepBuffer(self.nstep, self.gamma)
        self.replay = ReplayMemory(self.replay_size, self.alpha, self.beta, 1e-5, self.state_space_dim, self.is_conv)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.num_atoms, self.is_conv, self.use_distributional).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.num_atoms, self.is_conv, self.use_distributional).to(self.device)
        
        self._update_target_net()

        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        # self.optim = torch.optim.RMSprop(self.qnet.parameters(), lr=self.lr, momentum=0.95)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False, map_location=self.device)
            self.qnet.load_state_dict(checkpoint["qnet"])
            self.target_qnet.load_state_dict(checkpoint["target_qnet"])
            self.optim.load_state_dict(checkpoint["optim"])

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            if self.use_distributional:
                # distributional RL action selection
                distributions = self.qnet(states.to(self.device)) # (num_envs, action_dim, num_atoms)
                probs = torch.softmax(distributions, dim=-1)
                qvals = torch.sum(probs * torch.from_numpy(self.atoms).to(self.device), dim=-1)
                actions = qvals.argmax(dim=-1)
                return actions
            else:
                # regular expected RL action selection
                q_values = self.qnet(states.to(self.device))
                actions = q_values.argmax(dim=-1)
                return actions
        
    def expected_update(self, all_s, all_a, all_G, all_sprime, all_done, sampled_nodes, is_weights):
        masks = 1 - all_done # (minibatch_size,)

        q_vals = self.qnet(all_s) # (minibatch_size, action_space_dim,)
        chosen_q_vals = q_vals.gather(1, all_a.unsqueeze(1)).squeeze(1) # (minibatch_size,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            greedy_action_selection = self.qnet(all_sprime).argmax(dim=-1) # (minibatch_size,)
            greedy_action_evaluation = self.target_qnet(all_sprime).gather(-1, greedy_action_selection.unsqueeze(1)).squeeze(1)
            targets = all_G + (self.gamma ** self.nstep) * greedy_action_evaluation * masks # (minibatch_size,)

        td_error = targets - chosen_q_vals # (minibatch_size)
        self.replay.update_priorities(sampled_nodes, td_error.detach().cpu().numpy())

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()

        loss = torch.nn.functional.mse_loss(chosen_q_vals, targets, reduction="none") # scalar
        # loss = (is_weights * loss).mean()
        loss = loss.mean()
        loss.backward()

        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cgn)

        self.optim.step()

        self.logger.gradient_step_complete(["qnet_loss"], [loss.item()])
        self.logger.network_update({"qnet":self.qnet.state_dict(), "target_qnet":self.target_qnet.state_dict(), "optim":self.optim.state_dict()})

        
    def distributional_update(self, all_s, all_a, all_G, all_sprime, all_done, sampled_nodes, is_weights):
        masks = 1 - all_done # (minibatch_size,)

        with torch.no_grad():
            original_target_next_dist = self.target_qnet(all_sprime) # (minibatch_size, action_dim, num_atoms) LOGITS
            original_next_dist = self.qnet(all_sprime) # (minibatch_size, action_dim, num_atoms) LOGITS

            original_next_probs = torch.softmax(original_next_dist, dim=-1)
            original_next_qvals = torch.sum(original_next_probs * torch.from_numpy(self.atoms).to(self.device), dim=-1) # (minibatch_size, action_dim)
            next_actions = original_next_qvals.argmax(dim=-1) # (minibatch_size,)

            next_best_dist = original_target_next_dist[range(self.minibatch_size), next_actions] # (minibatch_size, num_atoms) 
            next_best_probs = torch.softmax(next_best_dist, dim=-1) # (minibatch_size, num_atoms)

        # allocate space for the projected distribution
        projected = torch.zeros((self.minibatch_size, self.num_atoms), dtype=torch.float32).to(self.device)

        # for every atom
        for j in range(0, self.num_atoms):

            # project Tz_j onto the support z_i

            # apply distributional bellman update and clip into vmin vmax
            zj = self.vmin + j * self.delta_z # scalar
            Tzj = all_G + (self.gamma ** self.nstep) * zj * masks # (minibatch_size)
            Tzj.clamp_(self.vmin, self.vmax) # (minibatch_size,)

            # calculate the fractional index of Tzj on the N grid from vmin to vmax
            bj = (Tzj - self.vmin) / self.delta_z # (minibatch_size)

            # projected atom Tzj could be between l and u
            # so distribute the probability p_j between these
            # based on their distances (u - bj) and (bj - l)
            l = torch.floor(bj).long() # (minibatch_size)
            u = torch.ceil(bj).long() # (minibatch_size)

            # distribute probability of Tz_j to immediate neighbours
            projected[range(self.minibatch_size), l] += next_best_probs[:, j] * (u - bj)
            projected[range(self.minibatch_size), u] += next_best_probs[:, j] * (bj - l)

            # edge case if bj is an int then l==u and both terms above will be 0
            # in that case manually give all p_j to either neighbour
            equal_transitions = (l == u)
            projected[range(self.minibatch_size), l] += next_best_probs[:, j] * equal_transitions.float()

        # get the log probs of the distribution for the states s
        current_dist = self.qnet(all_s) # (minibatch_size, action_dim, num_atoms)
        chosen_dists = current_dist[range(self.minibatch_size), all_a]
        current_dist_log_probs = torch.log_softmax(chosen_dists, dim=-1) # (minibatch_size, action_dim)

        # calculate KL divergence loss
        kl_loss = -current_dist_log_probs * projected
        kl_loss = kl_loss.sum(dim=-1)

        self.replay.update_priorities(sampled_nodes, kl_loss.detach().cpu().numpy())

        kl_loss = (kl_loss * is_weights).mean()

        self.optim.zero_grad()
        kl_loss.backward()
        self.optim.step()


    def _improve(self):
        if self.replay.current_size < self.minibatch_size: return

        (all_s, all_a, all_G, all_sprime, all_done), sampled_nodes, importance_sampling_weights = self.replay.sample(self.minibatch_size) # (minibatch_size,)
        
        all_s = all_s.to(self.device, non_blocking=True) # (minibatch_size, state_space_dim)
        all_a = all_a.to(self.device, non_blocking=True) # (minibatch_size,) int64 for indexing!
        all_G = all_G.to(self.device, non_blocking=True) # (minibatch_size,)
        all_sprime = all_sprime.to(self.device, non_blocking=True) # (minibatch_size, state_space_dim)
        all_done = all_done.to(self.device, non_blocking=True) # (minibatch_size,)
        masks = 1.0 - all_done # (minibatch_size,)
        importance_sampling_weights = torch.from_numpy(importance_sampling_weights).to(self.device)

        if self.use_distributional:
            self.distributional_update(all_s, all_a, all_G, all_sprime, all_done, sampled_nodes, importance_sampling_weights)
        else:
            self.expected_update(all_s, all_a, all_G, all_sprime, all_done, sampled_nodes, importance_sampling_weights)

    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None):
        assert env.num_envs == 1
        assert utils.is_space_discrete(env.get_single_action_space())

        utils.seed(seed)
        self.logger = logger
        total_iterations = total_timesteps // self.update_freq
        current_game_states = torch.from_numpy(env.start_states).float()

        self._setup(env)

        for iteration in range(1, total_iterations + 1):

            for current_t in range(self.update_freq):

                self.logger.timestep_complete()

                if self.logger.timesteps_completed % self.C == 0:
                    self._update_target_net()
                self.beta = self.beta_scheduler.step()
                self.replay.beta = self.beta
            
                current_actions = self._get_actions(current_game_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)
                        
                current_rewards = torch.from_numpy(current_rewards).float()
                current_sprimes = torch.from_numpy(current_sprimes).float()
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float()

                self.nstep_buffer.add(
                    current_game_states.type(torch.uint8 if self.is_conv else torch.float32).detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards.detach().cpu(),
                    current_sprimes.type(torch.uint8 if self.is_conv else torch.float32).detach().cpu(),
                    current_dones.detach().cpu(),
                )

                if self.nstep_buffer.is_full():
                    self.replay.add(self.nstep_buffer.nsteps_done())

                current_game_states = current_sprimes
                
                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve()

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        