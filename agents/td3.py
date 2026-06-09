import random
from collections import deque
import torch

import agents
import envs
import utils

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        self.main_body = torch.nn.Sequential(
            torch.nn.Linear(*input_size, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, *output_size),
        )

    def forward(self, inp):
        main_body_out = self.main_body(inp)
        return main_body_out

class QFunc(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(QFunc, self).__init__()

        self.fc_nn = torch.nn.Sequential(
            torch.nn.Linear(input_size[0] + output_size[0], 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1),
        )

    def forward(self, inp):
        return self.fc_nn(inp)

class TD3(agents.Agent):

    def __init__(self, lr, gamma, noise_factor, replay_size, minibatch_size, target_factor, d, noise_clip, warmup_steps):
        self.lr = lr
        self.gamma = gamma
        self.noise_factor = noise_factor
        self.replay_size = replay_size
        self.minibatch_size = minibatch_size
        self.target_factor = target_factor
        self.d = d
        self.noise_clip = noise_clip
        self.warmup_steps = warmup_steps
        self.update_freq = 1
        self.device = torch.device("cpu")

    def _clamp_actions(self, actions: torch.Tensor):
        return actions.clamp(self.action_space_low, self.action_space_high)

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            actions = self.actor(states) # (num_envs, action_space_dim,)
            return self._clamp_actions(actions + torch.randn_like(actions) * self.noise_factor) # (num_envs, action_space_dim,)

    def _setup(self, env: envs.Environment):
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        self.action_space_high = torch.from_numpy(env.get_single_action_space().high).float().to(self.device)
        self.action_space_low = torch.from_numpy(env.get_single_action_space().low).float().to(self.device)

        self.actor = Actor(self.state_space_dim, self.action_space_dim).to(self.device)
        self.qfunc1 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)
        self.qfunc2 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_actor = Actor(self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_qfunc1 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_qfunc2 = QFunc(self.state_space_dim, self.action_space_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_qfunc1.load_state_dict(self.qfunc1.state_dict())
        self.target_qfunc2.load_state_dict(self.qfunc2.state_dict())

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qfunc1_optimiser = torch.optim.Adam(self.qfunc1.parameters(), lr=self.lr)
        self.qfunc2_optimiser = torch.optim.Adam(self.qfunc2.parameters(), lr=self.lr)

        self.replay = deque(maxlen=self.replay_size)

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
            target_policy_actions = self.target_actor(all_sprime) # (minibatch_size, action_space_dim,)
            target_policy_actions += (torch.randn_like(target_policy_actions) * self.noise_factor).clamp(-self.noise_clip, self.noise_clip) # (minibatch_size, action_space_dim,)
            target_policy_actions = self._clamp_actions(target_policy_actions)
            target_policy_network_input = torch.concat([all_sprime, target_policy_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)

            min_qvals = torch.min(self.target_qfunc1(target_policy_network_input), self.target_qfunc2(target_policy_network_input)).squeeze(1) # (minibatch_size,)
            targets = all_r + self.gamma * masks * min_qvals # (minibatch_size,)

        self.qfunc1_optimiser.zero_grad()
        qfunc1_loss = torch.nn.functional.mse_loss(qfunc1_vals, targets)
        qfunc1_loss.backward()
        self.qfunc1_optimiser.step()

        self.qfunc2_optimiser.zero_grad()
        qfunc2_loss = torch.nn.functional.mse_loss(qfunc2_vals, targets)
        qfunc2_loss.backward()
        self.qfunc2_optimiser.step()

        updating_actor = False
        if self.logger.timesteps_completed % self.d == 0:
            updating_actor = True
            # if there has been a big enough DELAY

            # update actor 

            policy_actions = self.actor(all_s) # (minibatch_size, action_space_dim,)
            policy_network_input = torch.concat([all_s, policy_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
            actor_loss = -self.qfunc1(policy_network_input).mean() # scalar

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_optimiser.step()

            # update target nets
            for target_param, param in zip(self.target_qfunc1.parameters(), self.qfunc1.parameters()):
                target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)
            
            for target_param, param in zip(self.target_qfunc2.parameters(), self.qfunc2.parameters()):
                target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)

        if updating_actor:
            self.logger.gradient_step_complete(["qfunc1_loss", "qfunc2_loss", "actor_loss"], [qfunc1_loss.item(), qfunc2_loss.item(), actor_loss.item()])
        else:
            self.logger.gradient_step_complete(["qfunc1_loss", "qfunc2_loss"], [qfunc1_loss.item(), qfunc2_loss.item()])

        self.logger.network_update({"actor":self.actor.state_dict(), "target_actor":self.target_actor.state_dict(), "qfunc1":self.qfunc1.state_dict(), "qfunc2":self.qfunc2.state_dict(), "target_qfunc1":self.target_qfunc1.state_dict(), "target_qfunc2":self.target_qfunc2.state_dict(), "actor_optimiser":self.actor_optimiser.state_dict(), "qfunc1_optimiser":self.qfunc1_optimiser.state_dict(), "qfunc2_optimiser":self.qfunc2_optimiser.state_dict()})

    def learn(self, total_timesteps: int, env: envs.Gymenv, logger: utils.Logger, seed: int = None, quiet: bool = False):
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
                    current_game_states.detach(),
                    current_actions.detach(),
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