from collections import deque
import random

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class DRQN(nn.Module):
    def __init__(self, conv, state_space_dim, action_space_dim):
        super(DRQN, self).__init__()
        self.conv = conv

        self.conv_output_dim = 3136 # hardcoded i am lazy

        self.conv_nn = nn.Sequential(
            nn.Conv2d(state_space_dim[0], 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(), # (3136,)
        )

        self.lstm = nn.LSTM(self.conv_output_dim, 512, batch_first=True)

        self.fc = nn.Linear(512, action_space_dim)
        
    def forward(self, input, input_hidden=None):
        # input is (batch_size (B), sequence_length (T), channels (C), height (H), width (W))
        # e.g. (32, T, 1, 84, 84)

        batch_size, seq_len, channels, height, width = input.shape
        norm_input = input / 255.0

        # merge batch/time for conv layer
        conv_out = self.conv_nn(norm_input.view(batch_size * seq_len, channels, height, width))

        # restore batch/time for lstm layer (with conv output)
        lstm_out, hidden = self.lstm(conv_out.view(batch_size, seq_len, self.conv_output_dim), input_hidden)

        qvals = self.fc(lstm_out)

        return qvals, hidden

class DRQNAgent(Agent):
    def __init__(self, device, logger, lr, conv, replay_memory_size, replay_warmup_length,
                 C, minibatch_size, gamma, unroll_iterations,
                 epsilon_start, epsilon_end, epsilon_decay_steps, 
                 clip_grad_norm, update_freq, 
                 load_nn_path=None, save_nn=False, job_title="drqn"):
        
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
        self.unroll_iterations = unroll_iterations
        self.update_freq = update_freq

        self.epsilon = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end 
        self.epsilon_decay_steps = epsilon_decay_steps 

        self.load_nn_path = load_nn_path
        self.save_nn = save_nn

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

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
        self.reward_history = []

        # create DQN, target DQN, optimiser, replay memory
        self.dqn = DRQN(self.conv, self.state_space_dim, self.action_space_dim).to(self.device)
        self.target_dqn = DRQN(self.conv, self.state_space_dim, self.action_space_dim).to(self.device)
        self.clone_qnet()
        self.optimiser = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.replay = deque(maxlen=self.replay_memory_size)

        self.game_hidden_states = (torch.zeros((1, self.num_envs, 512)).to(self.device), torch.zeros((1, self.num_envs, 512)).to(self.device))

        self.episode_storage = [[] for env in range(self.num_envs)]

        # load relevant models if necessary
        if self.load_nn_path is not None:
            checkpoint = torch.load(self.load_nn_path, weights_only=False, map_location=self.device)
            self.dqn.load_state_dict(checkpoint["dqn"])
            self.target_dqn.load_state_dict(checkpoint["target_dqn"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.time_step = checkpoint["time_step"]
            self.epsilon = checkpoint["epsilon"]

    def episode_terminated(self, env_idx):
        # retrieve and reset episode from storage
        full_episode = self.episode_storage[env_idx]
        self.episode_storage[env_idx] = []
        self.replay.append(full_episode)

        self.game_hidden_states[0][:, env_idx, :] = 0.0
        self.game_hidden_states[1][:, env_idx, :] = 0.0

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
            s = torch.tensor(s, dtype=torch.float32).to(self.device)
            s = s.unsqueeze(1) # (num_envs, 1, state_space_dim,) add a fake time/length dim            
            qvals, hidden = self.dqn(s, self.game_hidden_states) # (num_envs, 1, action_space_dim,), (1, num_envs, 512)
            qvals = qvals.squeeze(1).cpu().numpy() # (num_envs, action_space_dim) remove the fake time/length dim
            self.game_hidden_states = hidden
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

        # draw a sample of minibatch_size episodes from replay memory and process them
        minibatch = random.sample(self.replay, self.minibatch_size)

        all_s, all_a, all_r, all_sprime, all_done = [], [], [], [], []

        for episode in minibatch:
            batch_s, batch_a, batch_r, batch_sprime, batch_done = zip(*episode)
            start = np.random.randint(0, len(episode) - self.unroll_iterations + 1)
            end = start + self.unroll_iterations

            all_s.append(np.array(batch_s[start:end]))
            all_a.append(np.array(batch_a[start:end]))
            all_r.append(np.array(batch_r[start:end]))
            all_sprime.append(np.array(batch_sprime[start:end]))
            all_done.append(np.array(batch_done[start:end]))

        all_s = torch.tensor(np.array(all_s), dtype=torch.float32).to(self.device) # (minibatch_size, unroll_iterations, state_space_dim,)
        all_a = torch.tensor(np.array(all_a), dtype=torch.int64).to(self.device) # (minibatch_size, unroll_iterations,)
        all_r = torch.tensor(np.array(all_r), dtype=torch.float32).to(self.device) # (minibatch_size, unroll_iterations,)
        all_sprime = torch.tensor(np.array(all_sprime), dtype=torch.float32).to(self.device) # (minibatch_size, unroll_iterations, state_space_dim,)
        all_done = torch.tensor(np.array(all_done), dtype=torch.float32).to(self.device) # (minibatch_size, unroll_iterations,)

        q_vals, hidden = self.dqn(all_s, None) # (minibatch_size, unroll_iterations, action_space_dim,)
        chosen_q_vals = q_vals.gather(2, all_a.unsqueeze(-1)).squeeze(-1) # (minibatch_size, unroll_iterations,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            target_qvals, target_hidden = self.target_dqn(all_sprime) # (minibatch_size, unroll_iterations, action_space_dim,)
            targets = all_r + self.gamma * target_qvals.max(2)[0] * (1 - all_done) # (minibatch_size, unroll_iterations,)

        self.optimiser.zero_grad()
        loss = F.smooth_l1_loss(chosen_q_vals, targets) # scalar
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.clip_grad_norm)
        self.optimiser.step()

        self.logger.log("loss", loss.item(), len(self.reward_history) + self.time_step)
        self.logger.log("avg_qval", chosen_q_vals.mean().item(), len(self.reward_history) + self.time_step)
        self.logger.save_logs()

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
            self.episode_storage[env_idx].append((s[env_idx], a[env_idx], r[env_idx], sprime[env_idx], done[env_idx]))

            # if the env episode ended, save/log/reset rewards
            if done[env_idx]:
                self.episode_terminated(env_idx)

        # if replay memory has enough samples
        if len(self.replay) >= self.replay_warmup_length and len(self.replay) >= self.minibatch_size:
            
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

        """