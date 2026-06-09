import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        self.main_body = torch.nn.Sequential(
            torch.nn.Linear(*input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )

        self.mu = torch.nn.Linear(64, *output_size)
        self.log_sigma = torch.nn.Linear(64, *output_size)

    def forward(self, inp):
        main_body_out = self.main_body(inp)
        mu = self.mu(main_body_out)
        log_sigma = self.log_sigma(main_body_out).clamp(-20, 2)
        sigma = log_sigma.exp()
        return mu, sigma

class QFunc(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(QFunc, self).__init__()

        self.fc_nn = torch.nn.Sequential(
            torch.nn.Linear(input_size[0] + output_size[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, inp):
        return self.fc_nn(inp)

class SAC(agents.Agent):

    def __init__(self, lr, gamma, replay_size, minibatch_size, update_freq, alpha_start, target_factor, warmup_steps):
        self.lr = lr
        self.gamma = gamma
        self.replay_size = replay_size
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.alpha_start = alpha_start
        self.alpha = alpha_start
        self.target_factor = target_factor
        self.warmup_steps = warmup_steps
        self.device = torch.device("cpu")

    def _scale_action(self, a):
        if isinstance(a, np.ndarray):
            return a * (self.action_space_high - self.action_space_low) / 2 + (self.action_space_high + self.action_space_low) / 2
        elif isinstance(a, torch.Tensor):
            return a * self.torch_action_space_scale + self.torch_action_space_bias

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            mu, sigma = self.actor(states) # (num_envs, action_space_dim,)
            dist = torch.distributions.Normal(mu, sigma)
            action = self._scale_action(torch.tanh(dist.rsample()).detach()) # (num_envs, action_space_dim,)
            return action # (num_envs, action_space_dim,)

    def _setup(self, env: envs.Environment):
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        self.action_space_high = env.get_single_action_space().high
        self.action_space_low = env.get_single_action_space().low
        self.torch_action_space_scale = torch.tensor((self.action_space_high - self.action_space_low) / 2, dtype=torch.float32).to(self.device)
        self.torch_action_space_bias = torch.tensor((self.action_space_high + self.action_space_low) / 2, dtype=torch.float32).to(self.device)

        self.replay = deque(maxlen=self.replay_size)
        self.actor = Actor(self.state_space_dim, self.action_space_dim).to(self.device)
        self.qfunc1 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)
        self.qfunc2 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_qfunc1 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_qfunc2 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)

        self.entropy_target = -self.action_space_dim[0] # paper says this should be -dim(A) (e.g. -6 for HalfCheetah)

        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(self.alpha_start)).to(self.device)).to(self.device)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.target_qfunc1.load_state_dict(self.qfunc1.state_dict())
        self.target_qfunc2.load_state_dict(self.qfunc2.state_dict())

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qfunc1_optimiser = torch.optim.Adam(self.qfunc1.parameters(), lr=self.lr)
        self.qfunc2_optimiser = torch.optim.Adam(self.qfunc2.parameters(), lr=self.lr)

    def _improve(self):
        if len(self.replay) < self.minibatch_size: return

        minibatch = random.sample(self.replay, self.minibatch_size)
        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch)
        
        all_s = torch.cat(all_s).to(self.device) # (minibatch_size, state_space_dim,)
        all_a = torch.cat(all_a).to(self.device) # (minibatch_size, action_space_dim,)
        all_r = torch.cat(all_r).to(self.device) # (minibatch_size,)
        all_sprime = torch.cat(all_sprime).to(self.device) # (minibatch_size, state_space_dim,)
        all_done = torch.cat(all_done).to(self.device) # (minibatch_size,)
        masks = 1 - all_done # (minibatch_size,)

        # update Q funcs
        
        qfunc1_vals = self.qfunc1(torch.concat([all_s, all_a], dim=1)).squeeze(1) # (minibatch_size,)
        qfunc2_vals = self.qfunc2(torch.concat([all_s, all_a], dim=1)).squeeze(1) # (minibatch_size,)

        with torch.no_grad():
            # prepare some "fresh" CURRENT/ONPOLICY actions based on sprimes
            fresh_mu, fresh_sigma = self.actor(all_sprime) # (minibatch_size, action_space_dim,)
            fresh_dists = torch.distributions.Normal(fresh_mu, fresh_sigma)
            fresh_raw_actions = fresh_dists.rsample() # (minibatch_size, action_space_dim,)
            fresh_actions = torch.tanh(fresh_raw_actions) # (minibatch_size, action_space_dim,)
            fresh_scaled_actions = self._scale_action(fresh_actions) # (minibatch_size, action_space_dim,)
            fresh_action_network_input = torch.concat([all_sprime, fresh_scaled_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
            fresh_actions_log_probs = fresh_dists.log_prob(fresh_raw_actions).sum(-1) - torch.log(1 - fresh_actions.pow(2) + 1e-6).sum(-1) # (minibatch_size,)
        
            # calculate Q targets
            min_qvals = torch.min(self.target_qfunc1(fresh_action_network_input), self.target_qfunc2(fresh_action_network_input)).squeeze(1) # (minibatch_size,)
            qfunc_targets = all_r + self.gamma * masks * (min_qvals - self.log_alpha.exp().detach() * fresh_actions_log_probs) # (minibatch_size,)

        # backprop + SGD for both qfuncs
        self.qfunc1_optimiser.zero_grad()
        self.qfunc2_optimiser.zero_grad()
        qfunc1_loss = torch.nn.functional.mse_loss(qfunc1_vals, qfunc_targets)
        qfunc2_loss = torch.nn.functional.mse_loss(qfunc2_vals, qfunc_targets)
        qfunc1_loss.backward()
        qfunc2_loss.backward()
        self.qfunc1_optimiser.step()
        self.qfunc2_optimiser.step()

        # update actor

        fresh_mu, fresh_sigma = self.actor(all_s) # (minibatch_size, action_space_dim,)
        fresh_dists = torch.distributions.Normal(fresh_mu, fresh_sigma)
        fresh_raw_actions = fresh_dists.rsample() # (minibatch_size, action_space_dim,)
        fresh_actions = torch.tanh(fresh_raw_actions) # (minibatch_size, action_space_dim,)
        fresh_scaled_actions = self._scale_action(fresh_actions) # (minibatch_size, action_space_dim,)
        fresh_action_network_input = torch.concat([all_s, fresh_scaled_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
        fresh_actions_log_probs = fresh_dists.log_prob(fresh_raw_actions).sum(-1) - torch.log(1 - fresh_actions.pow(2) + 1e-6).sum(-1) # (minibatch_size,)

        min_qvals = torch.min(self.qfunc1(fresh_action_network_input), self.qfunc2(fresh_action_network_input)).squeeze(1) # (minibatch_size,)

        self.actor_optimiser.zero_grad()
        policy_loss = -(min_qvals - self.log_alpha.exp().detach() * fresh_actions_log_probs).mean()
        policy_loss.backward()
        self.actor_optimiser.step()

        # update alpha

        alpha_loss = -(self.log_alpha * (fresh_actions_log_probs.detach() + self.entropy_target)).mean()
        
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()

        # update target qfuncs

        for target_param, param in zip(self.target_qfunc1.parameters(), self.qfunc1.parameters()):
            target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)
        
        for target_param, param in zip(self.target_qfunc2.parameters(), self.qfunc2.parameters()):
            target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)

        self.logger.gradient_step_complete(["qfunc1_loss", "qfunc2_loss", "policy_loss", "alpha_loss"], [qfunc1_loss.item(), qfunc2_loss.item(), policy_loss.item(), alpha_loss.item()])
        self.logger.network_update({
            "actor": self.actor.state_dict(),
            "qfunc1": self.qfunc1.state_dict(),
            "qfunc2": self.qfunc2.state_dict(),
            "target_qfunc1": self.target_qfunc1.state_dict(),
            "target_qfunc2": self.target_qfunc2.state_dict(),
            "actor_optimiser": self.actor_optimiser.state_dict(),
            "qfunc1_optimiser": self.qfunc1_optimiser.state_dict(),
            "qfunc2_optimiser": self.qfunc2_optimiser.state_dict(),
            "alpha_optimiser": self.alpha_optimiser.state_dict(),
        })

    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None, quiet: bool = False):
        assert env.get_num_envs() == 1
        assert utils.is_space_continuous(env.get_single_action_space())

        total_iterations = total_timesteps // self.update_freq
        utils.seed(seed)
        self.logger = logger
        current_game_states = torch.from_numpy(env.get_start_states()).float().to(self.device)

        self._setup(env)

        for iteration in range(1, total_iterations + 1):
            
            for current_t in range(self.update_freq):
                self.logger.timestep_complete()

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

                self.replay.append((
                    current_game_states,
                    current_actions,
                    current_rewards,
                    current_sprimes,
                    current_dones,
                ))

                current_game_states = current_sprimes

            if logger.timesteps_completed > self.warmup_steps:
                self._improve()
        
        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device