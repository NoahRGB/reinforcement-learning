import random
from collections import deque
import numpy as np
import torch

import agents
import envs
import utils

class QNet(torch.nn.Module):
    def __init__(self, input_size, output_size, conv):
        super(QNet, self).__init__()
        self.conv = conv

        if conv:
            self.conv_body = torch.nn.Sequential(
                torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(), # 3136
            )

            self.lstm = torch.nn.LSTM(3136, 512, batch_first=True)
            self.fc = torch.nn.Linear(512, output_size)

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
            conv_out = self.conv_body(norm_input.view(batch_size * seq_len, channels, height, width))

            # restore batch/time for lstm layer (with conv output)
            lstm_out, hidden = self.lstm(conv_out.view(batch_size, seq_len, 3136), inp_hidden)
            qvals_out = self.fc(lstm_out)

            return qvals_out, hidden
        
        lstm_out, hidden = self.lstm(inp, inp_hidden)
        qvals_out = self.fc(lstm_out)
        return qvals_out, hidden

class R2D2(agents.Agent):

    def __init__(self, lr, replay_size, C, update_freq, minibatch_size, 
                 gamma, epsilon_scheduler: utils.LinearScheduler, cgn, 
                 warmup_steps, gradient_steps, seq_len,
                 load_path=None):
        
        self.lr = lr
        self.replay_size = replay_size
        self.C = C
        self.update_freq = update_freq
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = epsilon_scheduler.get_value()
        self.cgn = cgn
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.device = torch.device("cpu")
        self.load_path = load_path

    def _update_target_net(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        
        self.sequence_buffer = deque(maxlen=self.seq_len)
        self.replay = deque(maxlen=self.replay_size//self.seq_len)
        self.qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)
        self.target_qnet = QNet(self.state_space_dim, self.action_space_dim, self.is_conv).to(self.device)

        self._update_target_net()

        self.optim = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        # self.optim = torch.optim.RMSprop(self.qnet.parameters(), lr=self.lr, momentum=0.95)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path, weights_only=False, map_location=self.device)
            self.qnet.load_state_dict(checkpoint["qnet"])
            self.target_qnet.load_state_dict(checkpoint["target_qnet"])
            self.optim.load_state_dict(checkpoint["optim"])

    def _get_actions(self, states: torch.Tensor, running_hidden_states: tuple):
        with torch.no_grad():
            states_input = states.unsqueeze(1) # (num_envs, 1, state_dim) add fake time/seq dim
            q_values, new_running_hidden_states = self.qnet(states_input, running_hidden_states)
            if np.random.random() >= self.epsilon:
                q_values = q_values.squeeze(1) # (num_envs, action_dim) remove fake time/seq dim
                actions = q_values.argmax(dim=-1)
                return actions, new_running_hidden_states
            else:
                return torch.tensor([np.random.choice(self.action_space_dim)], dtype=torch.int64), new_running_hidden_states
        
    def _improve(self):
        if len(self.replay) < self.minibatch_size: return

        minibatch = random.sample(self.replay, self.minibatch_size)

        all_s = torch.stack([torch.stack([t[0] for t in seq[0]]).view(self.seq_len, -1) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len, state_space_dim)
        all_a = torch.stack([torch.stack([t[1] for t in seq[0]]).view(self.seq_len) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len)
        all_r = torch.stack([torch.stack([t[2] for t in seq[0]]).view(self.seq_len) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len)
        all_sprime = torch.stack([torch.stack([t[3] for t in seq[0]]).view(self.seq_len, -1) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len, state_space_dim)
        all_done = torch.stack([torch.stack([t[4] for t in seq[0]]).view(self.seq_len) for seq in minibatch]).to(self.device) # (minibatch_size, seq_len)
        all_initial_hidden_states = (
            torch.cat([seq[1][0] for seq in minibatch], dim=1).to(self.device), # (minibatch_size, 1, 512)
            torch.cat([seq[1][1] for seq in minibatch], dim=1).to(self.device) # (minibatch_size, 1, 512)
        )

        masks = 1 - all_done # (minibatch_size, seq_len)

        q_vals, state_value_hidden_state_out = self.qnet(all_s, all_initial_hidden_states) # (minibatch_size, action_space_dim,)
        chosen_q_vals = q_vals.gather(-1, all_a.unsqueeze(-1)).squeeze(-1) # (minibatch_size,)

        # compute the target values (using the target DQN)
        with torch.no_grad():
            greedy_action_qvals, _ = self.qnet(all_sprime, state_value_hidden_state_out) # (minibatch_size,)
            greedy_action_selection = greedy_action_qvals.argmax(dim=-1) # (minibatch_size,)

            greedy_action_evaluation_qvals, _ = self.target_qnet(all_sprime, state_value_hidden_state_out)
            greedy_action_evaluation = greedy_action_evaluation_qvals.gather(-1, greedy_action_selection.unsqueeze(-1)).squeeze(-1) # (minibatch_size,)

            targets = all_r + self.gamma * greedy_action_evaluation * masks

        # zero grads, calculate loss, backprop, optimiser step
        self.optim.zero_grad()
        loss = torch.nn.functional.mse_loss(chosen_q_vals, targets) # scalar
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

        running_hidden_states = (
            torch.zeros((1, 1, 512)).to(self.device), 
            torch.zeros((1, 1, 512)).to(self.device)
        )

        seq_initial_hidden_states = (
            running_hidden_states[0].detach().clone(),
            running_hidden_states[1].detach().clone()
        )

        for iteration in range(1, total_iterations + 1):

            for current_t in range(self.update_freq):
                self.logger.timestep_complete()

                if self.logger.timesteps_completed % self.C == 0:
                    self._update_target_net()
                self.epsilon = self.epsilon_scheduler.step()
            
                current_actions, running_hidden_states = self._get_actions(current_game_states, running_hidden_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())
     
                current_rewards = torch.from_numpy(current_rewards).float().to(self.device)
                current_sprimes = torch.from_numpy(current_sprimes).float().to(self.device)
                current_dones = torch.from_numpy(current_isterms | current_istruncs).float().to(self.device)

                self.sequence_buffer.append((
                    current_game_states.detach().cpu(),
                    current_actions.detach().cpu(),
                    current_rewards.detach().cpu(),
                    current_sprimes.detach().cpu(),
                    current_dones.detach().cpu(),
                ))

                if len(self.sequence_buffer) == (self.seq_len // 2):
                    seq_initial_hidden_states = (
                        running_hidden_states[0].detach().clone(),
                        running_hidden_states[1].detach().clone()
                    )

                if len(self.sequence_buffer) == self.seq_len:
                    self.replay.append((self.sequence_buffer.copy(),
                        (
                            seq_initial_hidden_states[0].detach().cpu(), 
                            seq_initial_hidden_states[1].detach().cpu()
                        )
                    ))

                    for _ in range(self.seq_len // 2):
                        self.sequence_buffer.popleft()


                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)

                        self.sequence_buffer.clear()
                        
                        running_hidden_states = (
                            torch.zeros((1, 1, 512)).to(self.device), 
                            torch.zeros((1, 1, 512)).to(self.device)
                        )

                        seq_initial_hidden_states = (
                            running_hidden_states[0].detach().clone(),
                            running_hidden_states[1].detach().clone()
                        )

                current_game_states = current_sprimes
                
                if self.gradient_steps == -1 and self.logger.timesteps_completed > self.warmup_steps:
                    self._improve()

            if self.gradient_steps != -1 and self.logger.timesteps_completed > self.warmup_steps:
                for grad_update in range(self.gradient_steps):
                    self._improve()

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device
        