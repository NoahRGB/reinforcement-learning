import numpy as np
import torch

import agents
import envs
import utils

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, is_continuous: bool, is_conv: bool):
        super(ActorCriticNetwork, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_continuous = is_continuous
        self.is_conv = is_conv

        if is_conv:
            
            self.body_out_size = 3136
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )

        else:
            self.body_out_size = 64
            self.body = torch.nn.Sequential(
                torch.nn.Linear(*num_inputs, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh()
            )

        self.critic_head = torch.nn.Linear(self.body_out_size, 1)

        if is_continuous:
            self.mu_head = torch.nn.Linear(self.body_out_size, *num_outputs)
            self.log_sigma_head = torch.nn.Parameter(torch.zeros(*num_outputs))
        else:
            self.logits_head = torch.nn.Linear(self.body_out_size, num_outputs)

    def forward(self, inp: torch.Tensor):
        main_out = self.body(inp)
        critic_out = self.critic_head(main_out).squeeze(-1)
        if self.is_continuous:
            mu_out = self.mu_head(main_out)
            log_sigma_out = self.log_sigma_head
            return (mu_out, log_sigma_out.exp()), critic_out
        logits_out = self.logits_head(main_out)
        return logits_out, critic_out


class PPO(agents.Agent):
    def __init__(self, lr, gamma, lam, tmax, epsilon, epochs, minibatch_size, value_weight, entropy_weight, cgn):
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.tmax = tmax
        self.epsilon = epsilon
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.cgn = cgn
        self.device = torch.device("cpu")

    def _setup(self, env: envs.Environment):
        self.is_conv = env.is_conv()
        self.is_continuous = utils.is_space_continuous(env.get_single_action_space())
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        self.net = ActorCriticNetwork(self.state_space_dim, self.action_space_dim, self.is_continuous, self.is_conv).to(self.device)
        self.optim = torch.optim.RMSprop(self.net.parameters(), self.lr, eps=1e-5)

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            if self.is_continuous:
                (mu, sigma), _ = self.net(states)
                dist = torch.distributions.Normal(mu, sigma)
                actions = dist.sample()
                return dist, actions
            
            logits, _ = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            return dist, actions

    def _improve(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sprime: torch.Tensor, done: torch.Tensor, old_log_probs: torch.Tensor, num_envs: int):
        # s (tmax, num_envs, state_dim), a (tmax, num_envs, action_dim), r (tmax, num_envs), sprime (tmax, num_envs, state_dim), done (tmax, num_envs), old_log_probs (tmax, num_envs)
        masks = 1 - done # (tmax, num_envs)

        total_batch_size = self.tmax * num_envs
        all_indices = np.arange(total_batch_size)

        # for every epoch, shuffle the batch and step through it in minibatch chunks
        for epoch in range(self.epochs):

            # first calculate returns / advantages for the entire batch

            s_flattened = s.view(-1, *self.state_space_dim) # (tmax * num_envs, state_dim)
            sprime_flattened = sprime.view(-1, *self.state_space_dim) # (tmax * num_envs, state_dim)

            with torch.no_grad():
                _, state_values = self.net(s_flattened) # (tmax * num_envs, 1)
                state_values = state_values.view(self.tmax, num_envs) # (tmax, num_envs)

                _, next_state_values = self.net(sprime_flattened) # (tmax * num_envs, 1)
                next_state_values = next_state_values.view(self.tmax, num_envs) # (tmax, num_envs)

                gae = 0.0
                advantages = torch.zeros_like(r).to(self.device) # (tmax, num_envs)
                for t in reversed(range(self.tmax)):
                    delta = r[t] + self.gamma * next_state_values[t] * masks[t] - state_values[t]
                    gae = delta + self.gamma * self.lam * masks[t] * gae
                    advantages[t] = gae
            
            # now that advantages have been computed, all (tmax, num_envs) dims can be collapsed
            # since the temporal structure is no longer needed

            state_values = state_values.view(self.tmax * num_envs).detach() # (tmax * num_envs)
            advantages = advantages.view(self.tmax * num_envs).detach() # (tmax * num_envs,)
            s = s.view(self.tmax * num_envs, *s.shape[2:]) # (tmax * num_envs, state_dim)
            a = a.view(self.tmax * num_envs, *a.shape[2:]) # (tmax * num_envs, state_dim)
            old_log_probs = old_log_probs.view(self.tmax * num_envs,) # (tmax * num_envs)

            returns = (advantages + state_values).detach() # (tmax * num_envs)

            np.random.shuffle(all_indices)
            for minibatch_start_index in range(0, total_batch_size, self.minibatch_size):
                minibatch_indices = all_indices[minibatch_start_index : minibatch_start_index + self.minibatch_size]

                minibatch_s = s[minibatch_indices]
                minibatch_a = a[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                minibatch_old_log_probs = old_log_probs[minibatch_indices]

                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                if self.is_continuous:
                    (minibatch_mu, minibatch_sigma), minibatch_state_values = self.net(minibatch_s)
                    dist = torch.distributions.Normal(minibatch_mu, minibatch_sigma)
                    chosen_log_probs = dist.log_prob(minibatch_a).sum(-1) # (tmax * num_envs)
                else:
                    minibatch_logits, minibatch_state_values = self.net(minibatch_s)
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
        steps_per_iteration = self.tmax * env.get_num_envs()
        num_iterations = total_timesteps // steps_per_iteration

        self.logger = logger
        current_game_states = torch.from_numpy(env.get_start_states()).float().to(self.device)
        utils.seed(seed)

        self._setup(env)

        for iteration in range(1, num_iterations+1):

            # this iteration will require (tmax, num_envs) storage
            states = torch.zeros((self.tmax, env.get_num_envs(), *self.state_space_dim), dtype=torch.float32).to(self.device)
            actions = torch.zeros((self.tmax, env.get_num_envs()) + ((*self.action_space_dim,) if self.is_continuous else tuple()), dtype=torch.float32 if self.is_continuous else torch.int64).to(self.device)
            rewards = torch.zeros((self.tmax, env.get_num_envs()), dtype=torch.float32).to(self.device)
            sprimes = torch.zeros((self.tmax, env.get_num_envs(), *self.state_space_dim), dtype=torch.float32).to(self.device)
            dones = torch.zeros((self.tmax, env.get_num_envs()), dtype=torch.float32).to(self.device)
            old_log_probs = torch.zeros((self.tmax, env.get_num_envs()), dtype=torch.float32).to(self.device)
            
            for current_t in range(self.tmax):
                self.logger.timestep_complete(n=env.get_num_envs())

                dist, current_actions = self._get_actions(current_game_states)
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
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)

                current_game_states = sprimes[current_t]

            self._improve(states, actions, rewards, sprimes, dones, old_log_probs, env.num_envs)

        self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device