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
            nn.Linear(state_space_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, *action_space_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        main_body_out = self.main_body(input)
        return main_body_out

class QFunc(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(QFunc, self).__init__()

        self.fc_nn = nn.Sequential(
            nn.Linear(state_space_dim[0] + action_space_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input):
        return self.fc_nn(input)

class DDPGAgent(Agent):
    def __init__(self, device, logger, actor_lr, qfunc_lr, gamma, noise_factor,
                 replay_memory_size, minibatch_size, update_freq, target_factor,
                 decay_steps=None, decay_rate=0.99, save_nn=False, load_path=None, job_title="ddpg"):
        self.device = device
        self.logger = logger
        self.job_title = job_title
        self.actor_lr = actor_lr
        self.qfunc_lr = qfunc_lr
        self.replay_memory_size = replay_memory_size
        self.target_factor = target_factor
        self.noise_factor = noise_factor
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.gamma = gamma
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.save_nn = save_nn
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s, dtype=torch.float32).to(self.device)

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim)
        with torch.no_grad():
            actions = self.actor(self.process_state(s)) # (num_envs, action_space_dim,)
            return (actions + torch.randn_like(actions) * self.noise_factor).clamp(-1, 1).cpu().numpy() # (num_envs, action_space_dim,)

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs
        self.num_action_choices = action_space.num

        self.time_step = 0
        self.reward_record = -np.inf
                
        self.reward_history = []
        self.current_episode_rewards = np.zeros(num_envs)

        self.actor = Actor(self.state_space_size, self.action_space_size).to(self.device)
        self.qfunc = QFunc(self.state_space_size, self.action_space_size).to(self.device)
        self.target_actor = Actor(self.state_space_size, self.action_space_size).to(self.device)
        self.target_qfunc = QFunc(self.state_space_size, self.action_space_size).to(self.device)

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.qfunc_optimiser = optim.Adam(self.qfunc.parameters(), lr=self.qfunc_lr)

        self.replay = deque(maxlen=self.replay_memory_size)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.actor.load_state_dict(checkpoint["actor_nn"])
            self.actor_optimiser.load_state_dict(checkpoint["actor_optimiser"])
            self.qfunc.load_state_dict(checkpoint["qfunc_nn"])
            self.qfunc_optimiser.load_state_dict(checkpoint["qfunc_optimiser"])
            self.target_qfunc.load_state_dict(checkpoint["target_qfunc_nn"])
            self.target_actor.load_state_dict(checkpoint["target_actor_nn"])

    def update_qfuncs(self, all_s, all_a, all_r, all_sprime, all_done):
        masks = 1 - all_done # (minibatch_size,)

        qvals = self.qfunc(torch.concat([all_s, all_a], dim=1)).squeeze(1) # (minibatch_size,)

        with torch.no_grad():
            target_policy_actions = self.target_actor(all_sprime) # (minibatch_size, action_space_dim,)
            target_policy_network_input = torch.concat([all_sprime, target_policy_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
            targets = all_r + self.gamma * masks * self.target_qfunc(target_policy_network_input).squeeze(1) # (minibatch_size,)

        self.qfunc_optimiser.zero_grad()
        qfunc_loss = F.mse_loss(qvals, targets)
        qfunc_loss.backward()
        self.qfunc_optimiser.step()

    def update_actor(self, all_s):

        policy_actions = self.actor(all_s) # (minibatch_size, action_space_dim,)
        policy_network_input = torch.concat([all_s, policy_actions], dim=1) # (minibatch_size, state_space_dim + action_space_dim,)
        actor_loss = -self.qfunc(policy_network_input).mean() # scalar

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

    def update_target_qfuncs(self):
        for target_param, param in zip(self.target_qfunc.parameters(), self.qfunc.parameters()):
            target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.target_factor * target_param.data + (1 - self.target_factor) * param.data)

    def make_ddpg_update(self):
        # draw minibtach from replay memory and perform a DDPG update on the transitions
        minibatch = random.sample(self.replay, self.minibatch_size)

        all_s, all_a, all_r, all_sprime, all_done = zip(*minibatch)
        all_s = torch.tensor(np.array(all_s), dtype=torch.float32).to(self.device) # (minibatch_size, state_space_dim,)
        all_a = torch.tensor(np.array(all_a), dtype=torch.float32).to(self.device) # (minibatch_size,)
        all_r = torch.tensor(np.array(all_r), dtype=torch.float32).to(self.device) # (minibatch_size,)
        all_sprime = torch.tensor(np.array(all_sprime), dtype=torch.float32).to(self.device) # (minibatch_size, state_space_dim,)
        all_done = torch.tensor(np.array(all_done), dtype=torch.float32).to(self.device) # (minibatch_size,)

        self.update_qfuncs(all_s, all_a, all_r, all_sprime, all_done)
        self.update_actor(all_s)
        self.update_target_qfuncs()

    def update(self, s, sprime, a, r, done):

        # s is (num_envs, state_space_dim)
        # a is (num_envs, action_space_dim)
        # r is (num_envs,)
        # sprime is (num_envs, state_space_dim)
        # done is (num_envs,)

        self.time_step += 1
        self.current_episode_rewards += r

        for env_idx in range(self.num_envs):
            self.replay.append((s[env_idx], a[env_idx], r[env_idx], sprime[env_idx], done[env_idx]))

            if done[env_idx]:
                # if terminal, save/log/reset rewards, save model
                self.reward_history.append(self.current_episode_rewards[env_idx])
                mean_recent_reward = np.mean(self.reward_history[-100:])
                self.logger.log("mean_episode_reward", mean_recent_reward, step=len(self.reward_history))
                self.logger.log(f"episode_reward_{self.job_title}", self.current_episode_rewards[env_idx], step=len(self.reward_history))
                self.current_episode_rewards[env_idx] = 0.0

                if self.save_nn and mean_recent_reward > self.reward_record:
                    self.reward_record = mean_recent_reward
                    self.logger.save_torch({
                        "actor_nn": self.actor.state_dict(),
                        "actor_optimiser": self.actor_optimiser.state_dict(),
                        "qfunc_nn": self.qfunc.state_dict(),
                        "qfunc_optimiser": self.qfunc_optimiser.state_dict(),
                        "target_qfunc_nn": self.target_qfunc.state_dict(),
                    }, f"{self.job_title}_model")
                self.logger.save_logs()

        if len(self.replay) >= self.minibatch_size:
            if self.time_step % self.update_freq == 0:
                self.make_ddpg_update()
                print(f"time step: {self.time_step}")

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