import numpy as np
import torch

import agents
import envs
import utils

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, is_continuous: bool, is_conv: bool, lstm_hidden_size: int):
        super(ActorCriticNetwork, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_continuous = is_continuous
        self.is_conv = is_conv
        self.lstm_hidden_size = lstm_hidden_size
        self.body_out_size = 64

        if is_conv:
            
            self.conv_nn = torch.nn.Sequential(
                torch.nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(), # 3136
            )

            self.lstm = torch.nn.LSTM(3136, self.lstm_hidden_size, batch_first=False)
        else:

            self.actor_lstm = torch.nn.LSTM(*num_inputs, self.lstm_hidden_size, batch_first=False)
            self.critic_lstm = torch.nn.LSTM(*num_inputs, self.lstm_hidden_size, batch_first=False)

            self.actor_body = torch.nn.Sequential(
                torch.nn.Linear(self.lstm_hidden_size, self.body_out_size),
                torch.nn.ReLU(),
            )

            self.critic_body = torch.nn.Sequential(
                torch.nn.Linear(self.lstm_hidden_size, self.body_out_size),
                torch.nn.ReLU(),
            )


        self.critic_head = torch.nn.Linear(self.body_out_size, 1)

        if is_continuous:
            self.mu_head = torch.nn.Linear(self.body_out_size, *num_outputs)
            self.log_sigma_head = torch.nn.Parameter(torch.zeros(*num_outputs))
        else:
            self.logits_head = torch.nn.Linear(self.body_out_size, num_outputs)

    def forward(self, inp: torch.Tensor, inp_hidden: tuple = None):
        # inp is (batch_size (B), sequence_length (T), state_space_dim)

        if self.is_conv:
            batch_size, seq_len, channels, height, width = inp.shape
            norm_input = inp / 255.0
            conv_out = self.conv_nn(norm_input.view(batch_size * seq_len, channels, height, width))
            lstm_out, hidden_out = self.lstm(conv_out.view(seq_len, batch_size, 3136), inp_hidden)
            critic_lstm_out = lstm_out
            actor_lstm_out = lstm_out

        else:
            actor_inp_hidden, critic_inp_hidden = inp_hidden if inp_hidden is not None else (None, None)

            actor_lstm_out, actor_hidden_out = self.actor_lstm(inp, actor_inp_hidden)
            actor_lstm_out = self.actor_body(actor_lstm_out)

            critic_lstm_out, critic_hidden_out = self.critic_lstm(inp, critic_inp_hidden)
            critic_lstm_out = self.critic_body(critic_lstm_out)

            hidden_out = (actor_hidden_out, critic_hidden_out)
        
        critic_out = self.critic_head(critic_lstm_out).squeeze(-1)

        if self.is_continuous:
            mu_out = self.mu_head(actor_lstm_out)
            log_sigma_out = self.log_sigma_head
            return (mu_out, log_sigma_out.exp()), critic_out, hidden_out
        
        logits_out = self.logits_head(actor_lstm_out)

        return logits_out, critic_out, hidden_out


class LSTM_PPO(agents.Agent):
    def __init__(self, lr_scheduler: utils.LinearScheduler, gamma, lam, tmax, epsilon_scheduler: utils.LinearScheduler, epochs, minibatch_size, value_weight, entropy_weight, cgn, lstm_hidden_size):
        self.lr_scheduler = lr_scheduler
        self.lr = self.lr_scheduler.get_value()
        self.gamma = gamma
        self.lam = lam
        self.tmax = tmax
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = self.epsilon_scheduler.get_value()
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.cgn = cgn
        self.lstm_hidden_size = lstm_hidden_size
        self.device = torch.device("cpu")

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.is_continuous = utils.is_space_continuous(env.get_single_action_space())
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        self.net = ActorCriticNetwork(self.state_space_dim, self.action_space_dim, self.is_continuous, self.is_conv, self.lstm_hidden_size).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        # self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        # self.optim = torch.optim.RMSprop(self.net.parameters(), self.lr, eps=1e-5)

    def _get_actions(self, states: torch.Tensor, hidden_states: tuple):
        with torch.no_grad():
            states_inp = states.unsqueeze(0) # add fake time/seq dim

            if self.is_continuous:
                (mu, sigma), _, hidden_out = self.net(states_inp, hidden_states)
                mu = mu.squeeze(0) # remove the fake time/seq dim
                dist = torch.distributions.Normal(mu, sigma)
                actions = dist.sample()
                return dist, actions, hidden_out
            
            logits, _, hidden_out = self.net(states_inp, hidden_states)
            logits = logits.squeeze(0) # remove the fake time/seq dim
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()

            return dist, actions, hidden_out

    def _improve(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sprime: torch.Tensor, done: torch.Tensor, old_log_probs: torch.Tensor, num_envs: int, initial_hidden_states: tuple):
        # s (tmax, num_envs, state_dim)
        # a (tmax, num_envs, action_dim)
        # r (tmax, num_envs)
        # sprime (tmax, num_envs, state_dim)
        # done (tmax, num_envs)
        # old_log_probs (tmax, num_envs)
        
        masks = 1 - done # (tmax, num_envs)

        # compute all advantages at once for use in minibatches later
        with torch.no_grad():
            _, state_values, tmax_end_hidden = self.net(s, initial_hidden_states) # (tmax, num_envs)
            state_values = state_values.view(self.tmax, num_envs) # (tmax, num_envs)

            _, last_state_value, _ = self.net(sprime[-1].unsqueeze(0), tmax_end_hidden) # (1, num_envs)
            last_state_value = last_state_value.view(num_envs) # (num_envs,

            gae = 0.0
            advantages = torch.zeros_like(r).to(self.device) # (tmax, num_envs)
            for t in reversed(range(self.tmax)):
                if t == self.tmax - 1:
                    next_state_values = last_state_value * masks[t] # (num_envs)
                else:
                    next_state_values = state_values[t + 1] * masks[t] # (num_envs)
                delta = r[t] + self.gamma * next_state_values - state_values[t]
                gae = delta + self.gamma * self.lam * masks[t] * gae
                advantages[t] = gae
        

        returns = (advantages + state_values).detach() # (tmax, num_envs)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_env_indices = np.arange(num_envs)

        # for every epoch, shuffle the batch and step through it in minibatch chunks
        for epoch in range(self.epochs):
            np.random.shuffle(all_env_indices)
            for minibatch_start_index in range(0, num_envs, self.minibatch_size):
                minibatch_indices = all_env_indices[minibatch_start_index : minibatch_start_index + self.minibatch_size]

                # minibatch_state_values = state_values[minibatch_indices]
                minibatch_s = s[:, minibatch_indices, :] # (tmax, minibatch_size, state_dim)
                minibatch_a = a[:, minibatch_indices] # (tmax, minibatch_size, action_dim)
                minibatch_returns = returns[:, minibatch_indices] # (tmax, minibatch_size)
                minibatch_advantages = advantages[:, minibatch_indices] # (tmax, minibatch_size)
                minibatch_old_log_probs = old_log_probs[:, minibatch_indices] # (tmax, minibatch_size)

                if self.is_conv:
                    minibatch_hidden = (initial_hidden_states[0][:, minibatch_indices, :], initial_hidden_states[1][:, minibatch_indices, :])
                else:
                    minibatch_hidden = ((initial_hidden_states[0][0][:, minibatch_indices, :], initial_hidden_states[0][1][:, minibatch_indices, :]),
                                        (initial_hidden_states[1][0][:, minibatch_indices, :], initial_hidden_states[1][1][:, minibatch_indices, :]))

                if self.is_continuous:
                    (minibatch_mu, minibatch_sigma), minibatch_state_values, _ = self.net(minibatch_s, minibatch_hidden)
                    dist = torch.distributions.Normal(minibatch_mu, minibatch_sigma)
                    chosen_log_probs = dist.log_prob(minibatch_a).sum(-1) # (tmax * num_envs)
                else:
                    minibatch_logits, minibatch_state_values, _ = self.net(minibatch_s, minibatch_hidden)
                    log_probs = torch.nn.functional.log_softmax(minibatch_logits, dim=-1) # (tmax * num_envs, action_dim)
                    chosen_log_probs = log_probs.gather(-1, minibatch_a.unsqueeze(-1)).squeeze(-1) # (tmax * num_envs)
                    dist = torch.distributions.Categorical(logits=minibatch_logits)
                
                ratios = torch.exp(chosen_log_probs - minibatch_old_log_probs) # (tmax * num_envs)
                clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) # (tmax, num_envs)
                surrogate_obj = ratios * minibatch_advantages # (tmax * num_envs)
                clipped_surrogate_obj = clipped_ratios * minibatch_advantages # (tmax * num_envs)

                entropy_bonus = dist.entropy().mean()

                policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean() - (self.entropy_weight * entropy_bonus)

                state_value_loss = torch.nn.functional.mse_loss(minibatch_state_values, minibatch_returns)

                combined_loss = policy_loss + self.value_weight * state_value_loss

                self.optim.zero_grad()
                combined_loss.backward()
                if self.cgn is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cgn)
                self.optim.step()

                self.logger.gradient_step_complete(["policy_loss", "state_value_loss"], [policy_loss.item(), state_value_loss.item()])
    
        self.logger.network_update({"net": self.net.state_dict(), "optim": self.optim.state_dict()})

                
    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None, quiet: bool = False):
        # per iteration (tmax * num_envs) timesteps are unrolled
        # therefore to complete total_timesteps of learning
        # (total_timesteps / (tmax * num_envs)) iterations are needed
        num_envs = env.get_num_envs()
        steps_per_iteration = self.tmax * num_envs
        num_iterations = total_timesteps // steps_per_iteration

        self.logger = logger
        current_game_states = torch.from_numpy(env.get_start_states()).float().to(self.device)
        utils.seed(seed)

        self._setup(env)

        if self.is_conv:
            hidden_states = (
                (torch.zeros((1, num_envs, self.lstm_hidden_size)).to(self.device), torch.zeros((1, num_envs, self.lstm_hidden_size)).to(self.device))
            )
        else:
            hidden_states = (
                (torch.zeros((1, num_envs, self.lstm_hidden_size)).to(self.device), torch.zeros((1, num_envs, self.lstm_hidden_size)).to(self.device)),
                (torch.zeros((1, num_envs, self.lstm_hidden_size)).to(self.device), torch.zeros((1, num_envs, self.lstm_hidden_size)).to(self.device))
            ) # separate hidden states for actor + critic

        for iteration in range(1, num_iterations+1):

            # this iteration will require (tmax, num_envs) storage
            states = torch.zeros((self.tmax, num_envs, *self.state_space_dim), dtype=torch.float32).to(self.device)
            actions = torch.zeros((self.tmax, num_envs) + ((*self.action_space_dim,) if self.is_continuous else tuple()), dtype=torch.float32 if self.is_continuous else torch.int64).to(self.device)
            rewards = torch.zeros((self.tmax, num_envs), dtype=torch.float32).to(self.device)
            sprimes = torch.zeros((self.tmax, num_envs, *self.state_space_dim), dtype=torch.float32).to(self.device)
            dones = torch.zeros((self.tmax, num_envs), dtype=torch.float32).to(self.device)
            old_log_probs = torch.zeros((self.tmax, num_envs), dtype=torch.float32).to(self.device)

            if self.is_conv:
                initial_hidden_states = (hidden_states[0].clone(), hidden_states[1].clone())
            else:
                initial_hidden_states = ((hidden_states[0][0].clone(), hidden_states[0][1].clone()), (hidden_states[1][0].clone(), hidden_states[1][1].clone()))

            for current_t in range(self.tmax):
                self.logger.timestep_complete(n=num_envs)
                
                self.lr = self.lr_scheduler.step(n=num_envs)
                for param_group in self.optim.param_groups:
                    param_group['lr'] = self.lr
                self.epsilon = self.epsilon_scheduler.step(n=num_envs)

                dist, current_actions, hidden_states = self._get_actions(current_game_states, hidden_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())

                states[current_t] = current_game_states
                actions[current_t] = current_actions
                rewards[current_t] = torch.from_numpy(current_rewards).float().to(self.device)
                sprimes[current_t] = torch.from_numpy(current_sprimes).float().to(self.device)
                dones[current_t] = torch.from_numpy(current_isterms | current_istruncs).float().to(self.device)

                if self.is_continuous:
                    old_log_probs[current_t] = dist.log_prob(current_actions).sum(-1)  
                else:
                    old_log_probs[current_t] = dist.log_prob(current_actions)              

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]

                    # reset the running hidden state for finished environments
                    if self.is_conv:
                        hidden_states[0][:, done_idxs, :] = 0
                        hidden_states[1][:, done_idxs, :] = 0
                    else:
                        hidden_states[0][0][:, done_idxs, :] = 0
                        hidden_states[0][1][:, done_idxs, :] = 0
                        hidden_states[1][0][:, done_idxs, :] = 0
                        hidden_states[1][1][:, done_idxs, :] = 0

                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)

                current_game_states = sprimes[current_t]

            self._improve(states, actions, rewards, sprimes, dones, old_log_probs, num_envs, initial_hidden_states)

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device