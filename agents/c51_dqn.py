import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class QNet(torch.nn.Module):
    def __init__(self, input_size, action_dim, num_atoms, conv):
        super(QNet, self).__init__()
        self.conv = conv
        self.action_dim = action_dim
        self.num_atoms = num_atoms

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
                torch.nn.Linear(512, action_dim*num_atoms)
            )
        else:
            self.body = torch.nn.Sequential(
                torch.nn.Linear(*input_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_dim*num_atoms),
            )
    
    def forward(self, inp):
        batch, *state_dims = inp.shape

        if self.conv:
            new_inp = inp / 255.0
            all_out = self.body(new_inp)
            return all_out.view(batch, self.action_dim, self.num_atoms)
        
        all_out = self.body(inp)
        return all_out.view(batch, self.action_dim, self.num_atoms)
class C51DQN(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, gamma, epsilon_scheduler: utils.LinearScheduler, cgn, warmup_steps, gradient_steps, vmin, vmax, N, load_path=None):
        self.lr = lr
        self.replay_size = replay_size
        self.C = C
        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = epsilon_scheduler.get_value()
        self.cgn = cgn
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.device = torch.device("cpu")
        self.load_path = load_path
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = N
        self.delta_z = (self.vmax - self.vmin) / (self.num_atoms - 1)
        self.atoms = np.array([self.vmin + i * self.delta_z for i in range(self.num_atoms)])

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.replay = deque(maxlen=self.replay_size)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.num_atoms, self.is_conv).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.num_atoms, self.is_conv).to(self.device)
        
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
            if np.random.random() >= self.epsilon:
                distributions = self.qnet(states.to(self.device)) # (num_envs, action_dim, num_atoms)
                probs = torch.softmax(distributions, dim=-1)
                qvals = torch.sum(probs * torch.from_numpy(self.atoms).to(self.device), dim=-1)
                actions = qvals.argmax(dim=-1)
                return actions
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64).to(self.device)
        
    def _improve(self):
        if len(self.replay) < self.minibatch_size: return

        minibatch = random.sample(self.replay, self.minibatch_size)
        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch)
        
        all_s = torch.cat(all_s).to(self.device) # (minibatch_size, state_space_dim)
        all_a = torch.cat(all_a).to(self.device) # (minibatch_size,)
        all_r = torch.cat(all_r).to(self.device) # (minibatch_size,)
        all_sprime = torch.cat(all_sprime).to(self.device) # (minibatch_size, state_space_dim)
        all_done = torch.cat(all_done).to(self.device) # (minibatch_size,)
        masks = 1.0 - all_done # (minibatch_size,)

        original_next_dist = self.target_qnet(all_sprime) # (minibatch_size, action_dim, num_atoms) LOGITS
        original_next_probs = torch.softmax(original_next_dist, dim=-1)
        original_next_qvals = torch.sum(original_next_probs * torch.from_numpy(self.atoms).to(self.device), dim=-1) # (minibatch_size, action_dim)
        
        next_actions = original_next_qvals.argmax(dim=-1) # (minibatch_size,)
        next_best_dist = original_next_dist[range(self.minibatch_size), next_actions] # (minibatch_size, num_atoms) 
        next_best_probs = torch.softmax(next_best_dist, dim=-1) # (minibatch_size, num_atoms)

        # allocate space for the projected distribution
        projected = torch.zeros((self.minibatch_size, self.num_atoms), dtype=torch.float32).to(self.device)

        # for every atom
        for j in range(0, self.num_atoms):

            # project Tz_j onto the support z_i

            # apply distributional bellman update and clip into vmin vmax
            zj = self.vmin + j * self.delta_z # scalar
            Tzj = all_r + self.gamma * zj * masks # (minibatch_size)
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

        # get the log probs of the distribution for the states s
        current_dist = self.qnet(all_s) # (minibatch_size, action_dim, num_atoms)
        chosen_dists = current_dist[range(self.minibatch_size), all_a]
        current_dist_log_probs = torch.log_softmax(chosen_dists, dim=-1) # (minibatch_size, action_dim)

        # calculate KL divergence loss
        kl_loss = -current_dist_log_probs * projected
        kl_loss = kl_loss.sum(dim=-1).mean()

        self.optim.zero_grad()
        kl_loss.backward()
        self.optim.step()


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
                self.epsilon = self.epsilon_scheduler.step()
            
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

                self.replay.append((
                    current_game_states.detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards.detach().cpu(),
                    current_sprimes.detach().cpu(),
                    current_dones.detach().cpu(),
                ))

                current_game_states = current_sprimes
                
                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve()

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        