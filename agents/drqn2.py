import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class ReplayMemory:
    def __init__(self, size, state_space_dim):
        self.max_size = size
        self.current_size = 0
        self.next_insert = 0
        self.s = torch.zeros((self.max_size, *state_space_dim), dtype=torch.float32)
        self.a = torch.zeros((self.max_size), dtype=torch.int64)
        self.r = torch.zeros((self.max_size), dtype=torch.float32)
        self.sprime = torch.zeros((self.max_size, *state_space_dim), dtype=torch.float32)
        self.done = torch.zeros((self.max_size), dtype=torch.float32)

    def add(self, s, a, r, sprime, done):
        self.s[self.next_insert] = s.detach().cpu()
        self.a[self.next_insert] = a.detach().cpu()
        self.r[self.next_insert] = r.detach().cpu()
        self.sprime[self.next_insert] = sprime.detach().cpu()
        self.done[self.next_insert] = done.detach().cpu()

        self.next_insert = (self.next_insert + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def get_minibatch(self, minibatch_size, unroll_iterations):
        legal = min(self.current_size, self.max_size) - unroll_iterations
        start_idxs = (np.random.randint(0, legal, size=minibatch_size) + self.next_insert) % self.max_size
        sequence_idxs = torch.from_numpy(
            (start_idxs[:, None] + np.arange(unroll_iterations)) % self.max_size
        )
        
        return (
            self.s[sequence_idxs],
            self.a[sequence_idxs],
            self.r[sequence_idxs],
            self.sprime[sequence_idxs],
            self.done[sequence_idxs],
        )



class QNet(torch.nn.Module):
    def __init__(self, input_size, output_size, conv):
        super(QNet, self).__init__()
        self.conv = conv
        
        if conv:
            self.conv_nn = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(), # (3136,)
            )

            self.lstm = torch.nn.LSTM(3136, 512, batch_first=True)

        else:
            self.lstm = torch.nn.LSTM(*input_size, 512, batch_first=True)
 
        self.fc = torch.nn.Linear(512, output_size)
    
    def forward(self, inp, inp_hidden=None):
        # input is (batch_size (B), sequence_length (T), state_space_dim)
        # e.g. (32, T, 1, 84, 84)

        if self.conv:
            # input is (batch_size (B), sequence_length (T), channels (C), height (H), width (W))
            batch_size, seq_len, channels, height, width = inp.shape
            norm_input = inp / 255.0

            # merge batch/time for conv layer
            conv_out = self.conv_nn(norm_input.view(batch_size * seq_len, channels, height, width))

            # restore batch/time for lstm layer (with conv output)
            lstm_out, hidden = self.lstm(conv_out.view(batch_size, seq_len, 3136), inp_hidden)

            qvals = self.fc(lstm_out)

            return qvals, hidden

        else:
            lstm_out, hidden = self.lstm(inp, inp_hidden)

            qvals = self.fc(lstm_out)

            return qvals, hidden

class DRQN2(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, gamma, epsilon_start, epsilon_end, epsilon_steps, cgn, warmup_steps, unroll_iterations, load_path=None):
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
        self.unroll_iterations = unroll_iterations
        self.load_path = load_path
        self.device = torch.device("cpu")

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.replay = ReplayMemory(self.replay_size, self.state_space_dim)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)

        self.game_hidden_states = (torch.zeros((1, 1, 512)).to(self.device), torch.zeros((1, 1, 512)).to(self.device))
        
        self._update_target_net()

        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.epsilon_scheduler = utils.LinearScheduler(self.epsilon_start, self.epsilon_end, self.epsilon_steps)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False, map_location=self.device)
            self.qnet.load_state_dict(checkpoint["qnet"])
            self.target_qnet.load_state_dict(checkpoint["target_qnet"])
            self.optim.load_state_dict(checkpoint["optim"])

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            if np.random.random() >= self.epsilon:
                states_input = states.unsqueeze(1) # (num_envs, 1, state_dim) add fake time/seq dim
                q_values, hidden = self.qnet(states_input, self.game_hidden_states)
                actions = q_values.squeeze(1).argmax(dim=-1)
                self.game_hidden_states = hidden
                return actions
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64).to(self.device)
        
    def _improve(self):
        if self.replay.current_size < self.minibatch_size: return

        minibatch = self.replay.get_minibatch(self.minibatch_size, self.unroll_iterations)
        all_s, all_a, all_r, all_sprime, all_done = minibatch
        all_s = all_s.to(self.device)
        all_a = all_a.to(self.device)
        all_r = all_r.to(self.device)
        all_sprime = all_sprime.to(self.device)
        all_done = all_done.to(self.device)
        
        q_vals, hidden = self.qnet(all_s, None) # (minibatch_size, unroll_iterations, action_space_dim,)
        chosen_q_vals = q_vals.gather(2, all_a.unsqueeze(-1)).squeeze(-1) # (minibatch_size, unroll_iterations,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            target_qvals, target_hidden = self.target_qnet(all_sprime, None) # (minibatch_size, unroll_iterations, action_space_dim,)
            targets = all_r + self.gamma * target_qvals.max(2)[0] * (1 - all_done) # (minibatch_size, unroll_iterations,)

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(chosen_q_vals, targets) # scalar
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
            
                current_actions = self._get_actions(current_game_states.to(self.device))
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())
      
                current_rewards = torch.from_numpy(current_rewards).float()
                current_sprimes = torch.from_numpy(current_sprimes).float()
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float()

                self.replay.add(
                    current_game_states.detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards,
                    current_sprimes,
                    current_dones,
                )

                current_game_states = current_sprimes

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:

                        self.logger.episode_complete(reward)
                        self.game_hidden_states = (torch.zeros((1, 1, 512)).to(self.device), torch.zeros((1, 1, 512)).to(self.device))


            if self.logger.timesteps_completed > self.warmup_steps:
                self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        