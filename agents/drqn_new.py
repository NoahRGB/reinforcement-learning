import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class QNet(torch.nn.Module):
    def __init__(self, input_size, output_size, conv, lstm_size):
        super(QNet, self).__init__()
        self.conv = conv
        self.lstm_size = lstm_size
        
        if conv:
            self.lstm_out_size = 1024
            self.conv_nn = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 16, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )

            # self.lstm_out_size = 3136
            # self.conv_nn = torch.nn.Sequential(
            #     torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #     torch.nn.ReLU(),
            #     torch.nn.Flatten(), # (3136,)
            # )

            self.lstm = torch.nn.LSTM(self.lstm_out_size, self.lstm_size, batch_first=True)

        else:
            self.lstm = torch.nn.LSTM(*input_size, self.lstm_size, batch_first=True)
 
        self.fc = torch.nn.Linear(self.lstm_size, output_size)
    
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
            lstm_out, hidden = self.lstm(conv_out.view(batch_size, seq_len, self.lstm_out_size), inp_hidden)

            qvals = self.fc(lstm_out)

            return qvals, hidden

        else:
            lstm_out, hidden = self.lstm(inp, inp_hidden)

            qvals = self.fc(lstm_out)

            return qvals, hidden

class NewDRQN(agents.Agent):

    def __init__(self, lr_scheduler, replay_size, C, update_freq, 
                 minibatch_size, gamma, epsilon_scheduler, cgn, warmup_steps, 
                 gradient_steps, seq_len, overlap, lstm_size, load_path=None):
        self.lr_scheduler = lr_scheduler
        self.lr = lr_scheduler.get_value()
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
        self.seq_len = seq_len
        self.overlap = overlap
        self.load_path = load_path
        self.lstm_size = lstm_size
        self.device = torch.device("cpu")

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.logger.log_parameters(self)
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.sequence_buffer = deque(maxlen=self.seq_len)
        self.replay = deque(maxlen=self.replay_size//self.seq_len)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv, self.lstm_size).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv, self.lstm_size).to(self.device)
        
        self._update_target_net()
        self.sequence_buffer = deque(maxlen=self.seq_len)


        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False, map_location=self.device)
            self.qnet.load_state_dict(checkpoint["qnet"])
            self.target_qnet.load_state_dict(checkpoint["target_qnet"])
            self.optim.load_state_dict(checkpoint["optim"])
            if "norm" in checkpoint:
                env.load_normalised_obs(checkpoint["norm"])

    def _get_actions(self, states: torch.Tensor, running_hidden_states: tuple):
        with torch.no_grad():
            states_input = states.unsqueeze(1) # (num_envs, 1, state_dim) add fake time/seq dim
            q_values, hidden_out = self.qnet(states_input, running_hidden_states)
            if np.random.random() >= self.epsilon:
                actions = q_values.squeeze(1).argmax(dim=-1)
                return actions, hidden_out
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64).to(self.device), hidden_out
        
    def _improve(self, env: envs.Environment):
        if len(self.replay) < self.minibatch_size: return

        minibatch = random.sample(self.replay, self.minibatch_size)
        # print(f"minibatch len = {len(minibatch)}, minibatch type = {type(minibatch)}")
        # print(f"minibatch[0] len = {len(minibatch[0])}, minibatch[0] type = {type(minibatch[0])}")
        # print(f"minibatch[0][0] len = {len(minibatch[0][0])}, minibatch[0][0] type = {type(minibatch[0][0])}")

        # (minibatch_size, seq_len, state_dim)

        sequences = [list(tup[0]) for tup in minibatch] # (minibatch_size, seq_len, (s, a, r, s', done) )

        all_s = torch.stack([torch.stack([tup[0][0] for tup in sequence]) for sequence in sequences]).to(self.device) # (minibatch_size, seq_len, state_dim)
        all_a = torch.stack([torch.stack([tup[1][0] for tup in sequence]) for sequence in sequences]).to(self.device) # (minibatch_size, seq_len,)
        all_r = torch.stack([torch.stack([tup[2][0] for tup in sequence]) for sequence in sequences]).to(self.device) # (minibatch_size, seq_len,)
        all_sprime = torch.stack([torch.stack([tup[3][0] for tup in sequence]) for sequence in sequences]).to(self.device) # (minibatch_size, seq_len, state_dim)
        all_done = torch.stack([torch.stack([tup[4][0] for tup in sequence]) for sequence in sequences]).to(self.device) # (minibatch_size, seq_len,)
        all_initial_hidden_states = (
            torch.cat([seq[1][0] for seq in minibatch], dim=1).to(self.device), # (1, minibatch_size, lstm_size)
            torch.cat([seq[1][1] for seq in minibatch], dim=1).to(self.device) # (1, minibatch_size, lstm_size)
        )

        q_vals, hidden = self.qnet(all_s, None) # (minibatch_size, seq_len, action_space_dim,)
        chosen_q_vals = q_vals.gather(2, all_a.unsqueeze(-1)).squeeze(-1) # (minibatch_size, seq_len,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            target_qvals, target_hidden = self.target_qnet(all_sprime, None) # (minibatch_size, seq_len, action_space_dim,)
            targets = all_r + self.gamma * target_qvals.max(-1)[0] * (1 - all_done) # (minibatch_size, seq_len,)

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(chosen_q_vals, targets) # scalar
        loss.backward()
        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cgn)
        self.optim.step()

        self.logger.gradient_step_complete(["qnet_loss"], [loss.item()])
        log = {"qnet":self.qnet.state_dict(), "target_qnet":self.target_qnet.state_dict(), "optim":self.optim.state_dict()}
        if env.normalise_obs:
            log["norm"] = self.logger.env.get_normalised_obs()
        self.logger.network_update(log)

    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None):
        assert env.num_envs == 1
        assert utils.is_space_discrete(env.get_single_action_space())

        utils.seed(seed)
        self.logger = logger
        total_iterations = total_timesteps // self.update_freq
        current_game_states = torch.from_numpy(env.start_states).float()

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
                
                self.lr = self.lr_scheduler.step()
                for param in self.optim.param_groups:
                    param["lr"] = self.lr
            
                current_actions, new_running_hidden_states = self._get_actions(current_game_states.to(self.device), running_hidden_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())
      
                current_rewards = torch.from_numpy(current_rewards).float()
                current_sprimes = torch.from_numpy(current_sprimes).float()
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float()

                self.sequence_buffer.append((
                    current_game_states.detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards,
                    current_sprimes,
                    current_dones,
                    running_hidden_states,
                ))

                running_hidden_states = (
                    new_running_hidden_states[0].detach(),
                    new_running_hidden_states[1].detach(),
                )

                if len(self.sequence_buffer) == self.seq_len:
                    seq_initial_hidden_state = self.sequence_buffer[0][5]
                    self.replay.append((self.sequence_buffer.copy(), seq_initial_hidden_state))

                    for _ in range(self.overlap):
                        self.sequence_buffer.popleft()

                current_game_states = current_sprimes
                running_hidden_states = new_running_hidden_states

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:

                        self.logger.episode_complete(reward)
                        running_hidden_states = (
                            torch.zeros((1, 1, self.lstm_size)).to(self.device), 
                            torch.zeros((1, 1, self.lstm_size)).to(self.device)
                        )
                        self.sequence_buffer.clear()

                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve(env)

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve(env)

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        