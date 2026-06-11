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
        self.is_conv = is_conv
        self.is_continuous = is_continuous

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
            self.policy_body = torch.nn.Sequential(
                torch.nn.Linear(*num_inputs, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh()
            )

            self.value_body = torch.nn.Sequential(
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

        critic_out = self.critic_head(self.value_body(inp)).squeeze(-1)

        if self.is_continuous:
            mu_out = self.mu_head(self.policy_body(inp))
            log_sigma_out = self.log_sigma_head
            return (mu_out, log_sigma_out.exp()), critic_out
        
        logits_out = self.logits_head(self.policy_body(inp))
        return logits_out, critic_out


class A2C(agents.Agent):
    def __init__(self, lr: float, gamma: float, lam: float, tmax: int, value_weight: float, entropy_weight: float, cgn: float):
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.tmax = tmax
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
                return dist.sample()
            
            logits, _ = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            return actions
    
    def _improve(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sprime: torch.Tensor, done: torch.Tensor, num_envs: int):
        # s (tmax, num_envs, state_dim), a (tmax, num_envs, action_dim), r (tmax, num_envs), sprime (tmax, num_envs, state_dim), done (tmax, num_envs)

        s_flattened = s.view(-1, *self.state_space_dim) # (tmax * num_envs, state_dim)
        
        if self.is_continuous:
            (mu, sigma), state_values = self.net(s_flattened) # ( (tmax*num_envs, action_dim), (action_dim) ), (tmax*num_envs)
            mu = mu.view(self.tmax, num_envs, *self.action_space_dim)
        else:
            logits, state_values = self.net(s_flattened) # (tmax*num_envs, action_dim), (tmax*num_envs)
            logits = logits.view(self.tmax, num_envs, self.action_space_dim)

        state_values = state_values.view(self.tmax, num_envs) # (tmax, num_envs)
  
        _, next_state_values = self.net(sprime.view(-1, *self.state_space_dim)) # (tmax*num_envs, action_dim), (tmax*num_envs)
        next_state_values = next_state_values.view(self.tmax, num_envs) # (tmax, num_envs)

        masks = 1 - done # (tmax, num_envs)
        with torch.no_grad():
            gae = 0.0
            advantages = torch.zeros_like(r).to(self.device) # (tmax, num_envs)
            for t in reversed(range(len(r))):
                delta = r[t] + self.gamma * next_state_values[t] * masks[t] - state_values[t]
                gae = delta + self.gamma * self.lam * masks[t] * gae
                advantages[t] = gae

        returns = (advantages + state_values).detach() # (tmax, num_envs)

        if self.is_continuous:
            dist = torch.distributions.Normal(mu, sigma)
            chosen_log_probs = dist.log_prob(a).sum(-1) # (tmax, num_envs)
            entropy_bonus = dist.entropy().mean()
        else:
            log_probs = torch.log_softmax(logits, dim=-1) # (tmax, num_envs, action_dim,)
            chosen_log_probs = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1) # (tmax, num_envs, 1)
            entropy_bonus = torch.distributions.Categorical(logits=logits).entropy().mean()

        policy_loss = -(chosen_log_probs * advantages.detach()).mean() - (self.entropy_weight * entropy_bonus)
        state_value_loss = torch.nn.functional.mse_loss(state_values, returns)
        combined_loss = policy_loss + self.value_weight * state_value_loss

        self.optim.zero_grad()
        combined_loss.backward()
        if self.cgn is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cgn)
        self.optim.step()

        self.logger.gradient_step_complete(["policy_loss", "state_value_loss"], [policy_loss.item(), state_value_loss.item()])
        self.logger.network_update({"net":self.net.state_dict(), "optim":self.optim.state_dict()})

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

        for iteration in range(1, num_iterations + 1):
            # this iteration will require (tmax, num_envs) storage
            states = torch.zeros((self.tmax, env.get_num_envs(), *self.state_space_dim), dtype=torch.float32).to(self.device)
            actions = torch.zeros((self.tmax, env.get_num_envs()) + ((*self.action_space_dim,) if self.is_continuous else tuple()), dtype=torch.float32 if self.is_continuous else torch.int64).to(self.device)
            rewards = torch.zeros((self.tmax, env.get_num_envs()), dtype=torch.float32).to(self.device)
            sprimes = torch.zeros((self.tmax, env.get_num_envs(), *self.state_space_dim), dtype=torch.float32).to(self.device)
            dones = torch.zeros((self.tmax, env.get_num_envs()), dtype=torch.float32).to(self.device)

            for current_t in range(self.tmax):
                self.logger.timestep_complete(n=env.get_num_envs())

                current_actions = self._get_actions(current_game_states)
                current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())

                states[current_t] = current_game_states
                actions[current_t] = current_actions
                rewards[current_t] = torch.from_numpy(current_rewards).float().to(self.device)
                sprimes[current_t] = torch.from_numpy(current_sprimes).float().to(self.device)
                dones[current_t] = torch.from_numpy(current_isterms | current_istruncs).float().to(self.device)

                if "episode" in current_infos:
                    done_idxs = current_infos["_episode"]
                    completed_rewards = current_infos["episode"]["r"][done_idxs]
                    for reward in completed_rewards:
                        self.logger.episode_complete(reward)

                current_game_states = sprimes[current_t]

            self._improve(states, actions, rewards, sprimes, dones, env.get_num_envs())
        
        logger.training_done()

    def to(self, device: torch.device):
        self.device = device