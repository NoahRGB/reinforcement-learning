import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class IntrinsicCuriosityModule(torch.nn.Module):
    def __init__(self, state_dim, action_dim, is_conv):
        super(IntrinsicCuriosityModule, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_conv = is_conv

        if self.is_conv:
            self.enc_out = 1024
            self.enc = torch.nn.Sequential(
                torch.nn.Conv2d(state_dim[0], 16, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            # self.env_out = 3136
            # self.enc = torch.nn.Sequential(
            #     torch.nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #     torch.nn.ReLU(),
            #     torch.nn.Flatten(),
            # )
        else:
            self.enc_out = 64
            self.enc = torch.nn.Sequential(
                torch.nn.Linear(state_dim[0], 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.enc_out),
            )
        
        # takes φ(s_t) and φ(s_t+1) and predicts action a_t
        self.inv_model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.enc_out, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

        # takes a_t and φ(s_t) and predicts φ(s_t+1)
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(action_dim + self.enc_out, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.enc_out),
        )

        self.qnet = torch.nn.Linear(self.enc_out, action_dim)
        
    def forward(self, s, a, sprime):
        # s (batch_size, state_dim)
        # a (batch_size,)
        # sprime (batch_size, state_dim)

        a_one_hot = torch.nn.functional.one_hot(a, num_classes=self.action_dim).float()
        s_enc = self.enc(s)
        sprime_enc = self.enc(sprime)

        a_pred = self.inv_model(torch.concat([s_enc, sprime_enc], dim=1))
        sprime_enc_pred = self.forward_model(torch.concat([s_enc, a_one_hot], dim=1))

        return a_pred, sprime_enc_pred, sprime_enc

    def qvals(self, s):
        s_enc = self.enc(s)
        return self.qnet(s_enc)


class QNet(torch.nn.Module):
    def __init__(self, input_size, output_size, conv):
        super(QNet, self).__init__()
        self.conv = conv

        if conv:
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 16, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, output_size)
            )

            # self.body = torch.nn.Sequential(
            #     torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #     torch.nn.ReLU(),
            #     torch.nn.Flatten(),
            #     torch.nn.Linear(3136, 512),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(512, output_size)
            # )
        else:
            self.body = torch.nn.Sequential(
                torch.nn.Linear(*input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_size),
            )
    
    def forward(self, inp):
        if self.conv:
            new_inp = inp / 255.0
            return self.body(new_inp)
        return self.body(inp)

class CuriousDQN(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, gamma, epsilon_scheduler: utils.LinearScheduler, cgn, warmup_steps, gradient_steps, curiosity_weight, beta, lam, load_path=None):
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
        self.curiosity_weight = curiosity_weight
        self.beta = beta
        self.lam = lam
        self.device = torch.device("cpu")
        self.load_path = load_path

    def _update_target_net(self):
        # self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_icm.load_state_dict(self.icm.state_dict())

    def _setup(self, env: envs.Environment):
        self.logger.log_parameters(self)
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.replay = deque(maxlen=self.replay_size)

        self.icm = IntrinsicCuriosityModule(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)
        self.target_icm = IntrinsicCuriosityModule(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)
        self.icm_optim = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self._update_target_net()

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False, map_location=self.device)
            self.icm.load_state_dict(checkpoint["icm"])
            self.target_icm.load_state_dict(checkpoint["target_icm"])
            self.icm_optim.load_state_dict(checkpoint["icm_optim"])
            if "norm" in checkpoint:
                env.load_normalised_obs(checkpoint["norm"])

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            if np.random.random() >= self.epsilon:
                q_values = self.icm.qvals(states)
                actions = q_values.argmax(dim=-1)
                return actions
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64).to(self.device)
        
    def _improve(self, env: envs.Environment):
        if len(self.replay) < self.minibatch_size: return

        minibatch = random.sample(self.replay, self.minibatch_size)
        all_s, all_a, all_r, all_sprime, all_done, all_intrinsic_rewards = zip(*minibatch)
        
        all_s = torch.cat(all_s).to(self.device) # (minibatch_size, state_space_dim)
        all_a = torch.cat(all_a).to(self.device) # (minibatch_size,)
        all_r = torch.cat(all_r).to(self.device) # (minibatch_size,)
        all_sprime = torch.cat(all_sprime).to(self.device) # (minibatch_size, state_space_dim)
        all_done = torch.cat(all_done).to(self.device) # (minibatch_size,)
        all_intrinsic_rewards = torch.cat(all_intrinsic_rewards).to(self.device) # (minibatch_size,)
        masks = 1 - all_done # (minibatch_size,)

        a_pred, sprime_enc_pred, sprime_enc = self.icm(all_s, all_a, all_sprime)
        inv_loss = torch.nn.functional.cross_entropy(a_pred, all_a)
        forward_loss = torch.nn.functional.mse_loss(sprime_enc_pred, sprime_enc.detach(), reduction="none")

        # print(forward_loss.mean().item())

        all_r =  all_r + self.curiosity_weight * forward_loss.mean(-1) # (minibatch_size,)

        q_vals = self.icm.qvals(all_s) # (minibatch_size, action_space_dim,)
        chosen_q_vals = q_vals.gather(1, all_a.unsqueeze(1)).squeeze(1) # (minibatch_size,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            targets = all_r + self.gamma * self.target_icm.qvals(all_sprime).max(1)[0] * masks # (minibatch_size,)

        self.icm_optim.zero_grad()

        qnet_loss = torch.nn.functional.mse_loss(chosen_q_vals, targets) # scalar
        icm_loss = self.lam * qnet_loss + (1 - self.beta) * inv_loss + self.beta * forward_loss.mean()

        icm_loss.backward()

        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.cgn)

        self.icm_optim.step()

        self.logger.gradient_step_complete(["qnet_loss", "inv_loss", "forward_loss"], [qnet_loss.item(), inv_loss.item(), forward_loss.mean().item()])
        log = {"icm":self.icm.state_dict(), "target_icm":self.target_icm.state_dict(), "icm_optim":self.icm_optim.state_dict()}
        if env.normalise_obs:
            log["norm"] = env.get_normalised_obs()
        self.logger.network_update(log)


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

                with torch.no_grad():
                    a_pred, sprime_enc_pred, sprime_enc = self.icm(current_game_states, current_actions, current_sprimes)
                    intrinsic_reward = torch.nn.functional.mse_loss(sprime_enc_pred, sprime_enc, reduction="none").mean(-1)

                self.replay.append((
                    current_game_states.detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards.detach().cpu(),
                    current_sprimes.detach().cpu(),
                    current_dones.detach().cpu(),
                    intrinsic_reward
                ))

                current_game_states = current_sprimes
                
                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve(env)

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve(env)

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        