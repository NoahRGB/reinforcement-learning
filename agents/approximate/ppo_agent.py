from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import pickle

class Actor(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, conv, cont):
        super(Actor, self).__init__()
        self.conv = conv
        self.cont = cont

        if self.conv:
            self.fc_input_dim = 256
            self.main_body = nn.Sequential(
                nn.Conv2d(state_space_dim[0], 16, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2592, 256),
                nn.ReLU(),
            )
        else:
            self.fc_input_dim = 64
            self.main_body = nn.Sequential(
                nn.Linear(state_space_dim[0], 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )

        if self.cont:
            self.mu_nn = nn.Sequential(
                nn.Linear(self.fc_input_dim, *action_space_dim),
                nn.Tanh(),
            )

            self.log_std = nn.Parameter(torch.zeros(action_space_dim))

        else:
            self.logits_nn = nn.Sequential(
                nn.Linear(self.fc_input_dim, action_space_dim),
            )

    def forward(self, input):
        main_body_out = self.main_body(input)
        if self.cont:
            mu = self.mu_nn(main_body_out)
            sigma = torch.exp(self.log_std)
            sigma = torch.clamp(sigma, min=1e-3)
            return mu, sigma
        return self.logits_nn(main_body_out)

class Critic(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, conv):
        super(Critic, self).__init__()
        self.conv = conv

        if self.conv:
            self.fc_nn = nn.Sequential(
                nn.Conv2d(state_space_dim[0], 16, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2592, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
            self.fc_nn = nn.Sequential(
                nn.Linear(state_space_dim[0], 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

    def forward(self, input):
        return self.fc_nn(input)

class PPOAgent(Agent):
    def __init__(self, device, logger, actor_lr, critic_lr, gamma, lam, conv, cont,
                 epsilon, epochs, minibatch_size, tmax, entropy_weight, decay_steps=None, decay_rate=0.99, 
                 clip_grad_norm=None, save_nn=False, load_path=None, job_title="ppo"):
        self.device = device
        self.logger = logger
        self.job_title = job_title
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.tmax = tmax
        self.conv = conv
        self.cont = cont
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.entropy_weight = entropy_weight
        self.clip_grad_norm = clip_grad_norm
        self.save_nn = save_nn
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s, dtype=torch.float32).to(self.device)

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim)
        if self.cont:
            mu, sigma = self.actor(self.process_state(s)) # (num_envs, action_space_dim,)
            sigma = sigma.clamp(min=1e-3)
            dist = torch.distributions.Normal(mu, sigma)
            return dist.sample().cpu().numpy() # (num_envs, action_space_dim,)
        else:
            logits = self.actor(self.process_state(s)) # (num_envs, action_space_dim,)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample().cpu().numpy() # (num_envs, action_space_dim,)
    
    def reset_transitions(self):
        self.transitions = {
            "s": [],
            "a": [],
            "r": [],
            "sprime": [],
            "done": [],
            "log_probs": [],
        }

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs
        self.num_action_choices = action_space.num

        self.time_step = 0
        self.update_count = 0
        self.reward_record = -np.inf
        
        self.reset_transitions()
        
        self.reward_history = []
        self.current_episode_rewards = np.zeros(num_envs)

        self.actor = Actor(self.state_space_size, self.action_space_size, self.conv, self.cont).to(self.device)
        self.critic = Critic(self.state_space_size, self.action_space_size, self.conv).to(self.device)
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.actor.load_state_dict(checkpoint["actor_nn"])
            self.actor_optimiser.load_state_dict(checkpoint["actor_optimiser"])
            self.critic.load_state_dict(checkpoint["critic_nn"])
            self.critic_optimiser.load_state_dict(checkpoint["critic_optimiser"])

    def calculate_advantages(self):
        s = torch.tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        r = torch.tensor(np.array(self.transitions["r"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        sprime = torch.tensor(np.array(self.transitions["sprime"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        done = torch.tensor(np.array(self.transitions["done"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        masks = 1 - done # (tmax, num_envs)

        with torch.no_grad():
            gae = 0.0
            advantages = torch.zeros_like(r).to(self.device) # (tmax, num_envs)
            for t in reversed(range(len(r))):
                delta = r[t] + self.gamma * self.critic(sprime[t]).squeeze(-1) * masks[t] - self.critic(s[t]).squeeze(-1)
                gae = delta + self.gamma * self.lam * masks[t] * gae
                advantages[t] = gae

        advantages = advantages.view(self.tmax * self.num_envs).detach() # (tmax * num_envs,)
        return advantages

    def make_ppo_update(self, s, a, old_log_probs, advantages, returns):

        if self.cont:
            mu, sigma = self.actor(s) # (tmax*num_envs, action_space_dim)
            dist = torch.distributions.Normal(mu, sigma)
            chosen_log_probs = dist.log_prob(a).sum(-1) # (tmax*num_envs)
        else:
            logits = self.actor(s) # (tmax*num_envs, num_action_choices)
            log_probs = F.log_softmax(logits, dim=-1) # (tmax*num_envs, num_action_choices)
            chosen_log_probs = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1) # (tmax*num_envs)

        ratios = torch.exp(chosen_log_probs - old_log_probs) # (tmax*num_envs)
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) # (tmax*num_envs)
        surrogate_obj = ratios * advantages # (tmax*num_envs)
        clipped_surrogate_obj = clipped_ratios * advantages # (tmax*num_envs)

        if self.cont:
            entropy_bonus = dist.entropy().mean()
        else:
            entropy_bonus = torch.distributions.Categorical(logits=logits).entropy().mean()

        policy_loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean() - (self.entropy_weight * entropy_bonus)

        state_values = self.critic(s).squeeze(-1) # (tmax*num_envs)
        state_value_loss = F.mse_loss(state_values, returns)

        self.actor_optimiser.zero_grad()
        policy_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimiser.step()

        self.critic_optimiser.zero_grad()
        state_value_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.critic_optimiser.step()

        if self.decay_steps is not None:
            # self.scheduler.step()
            if self.time_step % self.decay_steps == 0:
                self.entropy_weight *= self.decay_rate

        self.logger.log("policy_loss", policy_loss.item(), self.update_count)
        self.logger.log("state_value_loss", state_value_loss.item(), self.update_count)
        self.logger.log("entropy", entropy_bonus.item(), self.update_count)
        if self.cont: self.logger.log("sigma", sigma.mean().item(), self.update_count)

        self.update_count += 1

    def update(self, s, sprime, a, r, done):

        # s is (num_envs, state_space_dim)
        # a is (num_envs, action_space_dim)
        # r is (num_envs,)
        # sprime is (num_envs, state_space_dim)
        # done is (num_envs,)

        self.transitions["s"].append(s)
        self.transitions["a"].append(a)
        self.transitions["r"].append(r)
        self.transitions["sprime"].append(sprime)
        self.transitions["done"].append(done)

        with torch.no_grad():
            if self.cont:
                mu, sigma = self.actor(self.process_state(s)) # (num_envs, action_space_dim)
                dist = torch.distributions.Normal(mu, sigma)
                chosen_log_probs = dist.log_prob(self.process_state(a)).sum(-1) # (num_envs,)
                self.transitions["log_probs"].append(chosen_log_probs.cpu().detach())
            else:
                logits = self.actor(self.process_state(s)) # (num_envs, action_space_dim)
                log_probs = F.log_softmax(logits, dim=-1) # (tmax, num_envs, num_action_choices)
                chosen_log_probs = log_probs[torch.arange(self.num_envs), a] # (num_envs,)
                self.transitions["log_probs"].append(chosen_log_probs.cpu().detach()) 

        self.time_step += 1
        self.current_episode_rewards += r

        if self.time_step % self.tmax == 0:
            advantages = self.calculate_advantages()

            s = torch.tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
            a = torch.tensor(np.array(self.transitions["a"]), dtype=torch.float32 if self.cont else torch.int64).to(self.device) # (tmax, num_envs)
            old_log_probs = torch.as_tensor(np.array(self.transitions["log_probs"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
            old_log_probs = old_log_probs.view(self.tmax * self.num_envs) # (tmax * num_envs,)
            s = s.view(self.tmax * self.num_envs, *s.shape[2:]) # (tmax * num_envs, state_space_dim)
            a = a.view(self.tmax * self.num_envs, *a.shape[2:]) # (tmax * num_envs, action_space_dim)

            state_values = self.critic(s).squeeze(-1) # (tmax * num_envs,)
            returns = (advantages + state_values).detach() # (tmax * num_envs,)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_batch_size = self.tmax * self.num_envs
            transition_indices = np.arange(total_batch_size, dtype=np.int32)

            for epoch in range(self.epochs):
                np.random.shuffle(transition_indices)
                for minibatch_start_index in range(0, total_batch_size, self.minibatch_size):
                    minibatch_indices = transition_indices[minibatch_start_index : minibatch_start_index + self.minibatch_size]
                    self.make_ppo_update(
                        s[minibatch_indices], 
                        a[minibatch_indices], 
                        old_log_probs[minibatch_indices],
                        advantages[minibatch_indices],
                        returns[minibatch_indices],
                    )

            self.reset_transitions()
            self.logger.save_logs()

        for env_idx in range(self.num_envs):
            if done[env_idx]:
                # if terminal, save/log/reset rewards, save model
                self.reward_history.append(self.current_episode_rewards[env_idx])
                mean_recent_reward = np.mean(self.reward_history[-100:])
                self.logger.log("mean_episode_reward", mean_recent_reward, len(self.reward_history))
                self.logger.log("episode_reward", self.current_episode_rewards[env_idx], len(self.reward_history))
                self.current_episode_rewards[env_idx] = 0.0

                if self.save_nn and mean_recent_reward > self.reward_record:
                    self.reward_record = mean_recent_reward
                    self.logger.save_torch({
                        "actor_nn": self.actor.state_dict(),
                        "actor_optimiser": self.actor_optimiser.state_dict(),
                        "critic_nn": self.critic.state_dict(),
                        "critic_optimiser": self.critic_optimiser.state_dict(),
                    }, f"{self.job_title}_model")
                self.logger.save_logs()

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