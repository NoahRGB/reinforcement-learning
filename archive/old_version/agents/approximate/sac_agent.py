import pickle, random

from collections import deque

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(Actor, self).__init__()

        self.fc_input_dim = 64
        self.main_body = nn.Sequential(
            nn.Linear(state_space_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, self.fc_input_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(self.fc_input_dim, *action_space_dim)
        self.log_sigma = nn.Linear(self.fc_input_dim, *action_space_dim)

    def forward(self, input):
        main_body_out = self.main_body(input)
        mu = self.mu(main_body_out)
        log_sigma = self.log_sigma(main_body_out).clamp(-20, 2)
        sigma = log_sigma.exp()
        return mu, sigma

class QFunc(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(QFunc, self).__init__()

        self.fc_nn = nn.Sequential(
            nn.Linear(state_space_dim[0] + action_space_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input):
        return self.fc_nn(input)

class SACAgent(Agent):
    def __init__(self, device, logger, lr, gamma, rand_steps,
                 replay_memory_size, minibatch_size, update_freq, alpha_start, target_factor,
                 decay_steps=None, decay_rate=0.99, save_nn=False, load_path=None, job_title="sac"):
        self.device = device
        self.logger = logger
        self.job_title = job_title
        self.lr = lr
        self.rand_steps = rand_steps
        self.replay_memory_size = replay_memory_size
        self.alpha_start = alpha_start
        self.target_factor = target_factor
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.gamma = gamma
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.save_nn = save_nn
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s, dtype=torch.float32).to(self.device)
    
    def scale_action(self, a):
        if isinstance(a, np.ndarray):
            return a * (self.action_space_high - self.action_space_low) / 2 + (self.action_space_high + self.action_space_low) / 2
        elif isinstance(a, torch.Tensor):
            return a * self.torch_action_space_scale + self.torch_action_space_bias

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim)
        if self.rand_mode:
            action = np.random.uniform(self.action_space_low, self.action_space_high, size=(self.num_envs, self.action_space_size[0]))
            return action
        
        mu, sigma = self.actor(torch.tensor(s, dtype=torch.float32).to(self.device)) # (num_envs, action_space_dim,)
        dist = torch.distributions.Normal(mu, sigma)
        action = self.scale_action(torch.tanh(dist.rsample()).detach().cpu().numpy()) # (num_envs, action_space_dim,)
        return action # (num_envs, action_space_dim,)

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.action_space_high = action_space.max_bounds
        self.action_space_low = action_space.min_bounds
        self.num_envs = num_envs

        self.torch_action_space_scale = torch.tensor((self.action_space_high - self.action_space_low) / 2, dtype=torch.float32).to(self.device)
        self.torch_action_space_bias = torch.tensor((self.action_space_high + self.action_space_low) / 2, dtype=torch.float32).to(self.device)

        if self.rand_steps > 0:
            self.rand_mode = True
        else:
            self.rand_mode = False

        self.time_step = 0
        self.gradient_updates = 0
        self.reward_record = -np.inf

        self.entropy_target = -self.action_space_size[0] # paper says this should be -dim(A) (e.g. -6 for HalfCheetah)

        self.logger.log("details", self.get_dump())

        self.reward_history = []
        self.current_episode_rewards = np.zeros(num_envs)

        self.actor = Actor(self.state_space_size, self.action_space_size).to(self.device)
        self.qfunc1 = QFunc(self.state_space_size, self.action_space_size).to(self.device)
        self.qfunc2 = QFunc(self.state_space_size, self.action_space_size).to(self.device)
        self.target_qfunc1 = QFunc(self.state_space_size, self.action_space_size).to(self.device)
        self.target_qfunc2 = QFunc(self.state_space_size, self.action_space_size).to(self.device)

        self.target_qfunc1.load_state_dict(self.qfunc1.state_dict())
        self.target_qfunc2.load_state_dict(self.qfunc2.state_dict())

        self.log_alpha = nn.Parameter(torch.tensor(np.log(self.alpha_start)).to(self.device)).to(self.device)
        self.alpha_optimiser = optim.Adam([self.log_alpha], lr=self.lr)

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qfunc1_optimiser = optim.Adam(self.qfunc1.parameters(), lr=self.lr)
        self.qfunc2_optimiser = optim.Adam(self.qfunc2.parameters(), lr=self.lr)

        self.replay = deque(maxlen=self.replay_memory_size)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.actor.load_state_dict(checkpoint["actor_nn"])
            self.actor_optimiser.load_state_dict(checkpoint["actor_optimiser"])
            self.qfunc1.load_state_dict(checkpoint["qfunc1_nn"])
            self.qfunc2.load_state_dict(checkpoint["qfunc2_nn"])
            self.qfunc1_optimiser.load_state_dict(checkpoint["qfunc1_optimiser"])
            self.qfunc2_optimiser.load_state_dict(checkpoint["qfunc2_optimiser"])
            self.target_qfunc1.load_state_dict(checkpoint["target_qfunc1_nn"])
            self.target_qfunc2.load_state_dict(checkpoint["target_qfunc2_nn"])

    def update_qfuncs(self, all_s, all_a, all_r, all_sprime, all_done):
        masks = 1 - all_done # (minibatch_size,)

        qfunc1_vals = self.qfunc1(torch.concat([all_s, all_a], dim=1)).squeeze(1) # (minibatch_size,)
        qfunc2_vals = self.qfunc2(torch.concat([all_s, all_a], dim=1)).squeeze(1) # (minibatch_size,)

        with torch.no_grad():
            # prepare some "fresh" CURRENT/ONPOLICY actions based on sprimes
            fresh_mu, fresh_sigma = self.actor(all_sprime) # (minibatch_size, action_space_dim,)
            fresh_dists = torch.distributions.Normal(fresh_mu, fresh_sigma)
            fresh_raw_actions = fresh_dists.rsample() # (minibatch_size, action_space_dim,)
            fresh_actions = torch.tanh(fresh_raw_actions) # (minibatch_size, action_space_dim,)
            fresh_scaled_actions = self.scale_action(fresh_actions) # (minibatch_size, action_space_dim,)
            fresh_action_network_input = torch.concat([all_sprime, fresh_scaled_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
            fresh_actions_log_probs = fresh_dists.log_prob(fresh_raw_actions).sum(-1) - torch.log(1 - fresh_actions.pow(2) + 1e-6).sum(-1) # (minibatch_size,)
        
            # calculate Q targets
            min_qvals = torch.min(self.target_qfunc1(fresh_action_network_input), self.target_qfunc2(fresh_action_network_input)).squeeze(1) # (minibatch_size,)
            qfunc_targets = all_r + self.gamma * masks * (min_qvals - self.log_alpha.exp().detach() * fresh_actions_log_probs) # (minibatch_size,)

        # backprop + SGD for both qfuncs
        self.qfunc1_optimiser.zero_grad()
        self.qfunc2_optimiser.zero_grad()
        qfunc1_loss = F.mse_loss(qfunc1_vals, qfunc_targets)
        qfunc2_loss = F.mse_loss(qfunc2_vals, qfunc_targets)
        qfunc1_loss.backward()
        qfunc2_loss.backward()
        self.qfunc1_optimiser.step()
        self.qfunc2_optimiser.step()

        self.logger.log("qfunc1_loss", qfunc1_loss.item(), self.gradient_updates)
        self.logger.log("qfunc2_loss", qfunc2_loss.item(), self.gradient_updates)
    
    def update_actor(self, all_s):
                    
        fresh_mu, fresh_sigma = self.actor(all_s) # (minibatch_size, action_space_dim,)
        fresh_dists = torch.distributions.Normal(fresh_mu, fresh_sigma)
        fresh_raw_actions = fresh_dists.rsample() # (minibatch_size, action_space_dim,)
        fresh_actions = torch.tanh(fresh_raw_actions) # (minibatch_size, action_space_dim,)
        fresh_scaled_actions = self.scale_action(fresh_actions) # (minibatch_size, action_space_dim,)
        fresh_action_network_input = torch.concat([all_s, fresh_scaled_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
        fresh_actions_log_probs = fresh_dists.log_prob(fresh_raw_actions).sum(-1) - torch.log(1 - fresh_actions.pow(2) + 1e-6).sum(-1) # (minibatch_size,)

        min_qvals = torch.min(self.qfunc1(fresh_action_network_input), self.qfunc2(fresh_action_network_input)).squeeze(1) # (minibatch_size,)

        self.actor_optimiser.zero_grad()
        policy_loss = -(min_qvals - self.log_alpha.exp().detach() * fresh_actions_log_probs).mean()
        policy_loss.backward()
        self.actor_optimiser.step()

        self.logger.log("policy_loss", policy_loss.item(), self.gradient_updates)
        return fresh_actions_log_probs

    def update_alpha(self, log_probs):

        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()
        
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()

    def update_target_qfuncs(self):
        for target_param, param in zip(self.target_qfunc1.parameters(), self.qfunc1.parameters()):
            target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)
        
        for target_param, param in zip(self.target_qfunc2.parameters(), self.qfunc2.parameters()):
            target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)

    def make_sac_update(self):
        # draw minibtach from replay memory and perform a SAC update on the transitions
        minibatch = random.sample(self.replay, self.minibatch_size)
        self.gradient_updates += 1

        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch)
        all_s = torch.tensor(np.array(all_s), dtype=torch.float32).to(self.device) # (minibatch_size, state_space_dim,)
        all_a = torch.tensor(np.array(all_a), dtype=torch.float32).to(self.device) # (minibatch_size,)
        all_r = torch.tensor(np.array(all_r), dtype=torch.float32).to(self.device) # (minibatch_size,)
        all_sprime = torch.tensor(np.array(all_sprime), dtype=torch.float32).to(self.device) # (minibatch_size, state_space_dim,)
        all_done = torch.tensor(np.array(all_done), dtype=torch.float32).to(self.device) # (minibatch_size,)

        self.update_qfuncs(all_s, all_a, all_r, all_sprime, all_done)
        log_probs = self.update_actor(all_s)
        self.update_alpha(log_probs)
        self.update_target_qfuncs()

    def update(self, s, sprime, a, r, done):

        # s is (num_envs, state_space_dim)
        # a is (num_envs, action_space_dim)
        # r is (num_envs,)
        # sprime is (num_envs, state_space_dim)
        # done is (num_envs,)

        self.time_step += self.num_envs
        self.current_episode_rewards += r

        for env_idx in range(self.num_envs):
            self.replay.append((s[env_idx], a[env_idx], r[env_idx], sprime[env_idx], done[env_idx]))

            if done[env_idx]:
                # if terminal, save/log/reset rewards, save model
                self.reward_history.append(self.current_episode_rewards[env_idx])
                mean_recent_reward = np.mean(self.reward_history[-100:])
                self.logger.log("mean_episode_reward", mean_recent_reward, step=self.time_step)
                self.logger.log(f"episode_reward_{self.job_title}", self.current_episode_rewards[env_idx], step=len(self.reward_history))
                self.current_episode_rewards[env_idx] = 0.0

                if self.save_nn and mean_recent_reward > self.reward_record:
                    self.reward_record = mean_recent_reward
                    self.logger.save_torch({
                        "actor_nn": self.actor.state_dict(),
                        "actor_optimiser": self.actor_optimiser.state_dict(),
                        "qfunc1_nn": self.qfunc1.state_dict(),
                        "qfunc2_nn": self.qfunc2.state_dict(),
                        "qfunc1_optimiser": self.qfunc1_optimiser.state_dict(),
                        "qfunc2_optimiser": self.qfunc2_optimiser.state_dict(),
                        "target_qfunc1_nn": self.target_qfunc1.state_dict(),
                        "target_qfunc2_nn": self.target_qfunc2.state_dict(),
                    }, f"{self.job_title}_model")
                self.logger.save_logs()

        self.rand_mode = (self.time_step < self.rand_steps)

        if len(self.replay) >= self.minibatch_size:
            if (self.time_step / self.num_envs) % self.update_freq == 0:
                self.make_sac_update()

    def finish_episode(self, episode_num):
        pass

    def get_supported_env_types(self):
        return [EnvType.SINGULAR, EnvType.VECTORISED]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace, ContinuousSpace]
    
    def get_dump(self):
        return f"""

        """