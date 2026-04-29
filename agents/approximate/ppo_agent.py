from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, conv, cont):
        super(Actor, self).__init__()
        self.conv = conv
        self.cont = cont

        if self.conv:
            self.fc_input_dim = 512
            self.main_body = nn.Sequential(
                nn.Conv2d(state_space_dim[0], 32, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.Flatten(), # (3136,)
                nn.Linear(3136, 512),
                nn.ReLU(),
            )
        else:
            self.fc_input_dim = 64
            self.main_body = nn.Sequential(
                nn.Linear(state_space_dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )

        if self.cont:
            self.mu_nn = nn.Sequential(
                nn.Linear(self.fc_input_dim, *action_space_dim),
                # nn.Tanh(),
            )

            self.sigma_nn = nn.Sequential(
                nn.Linear(self.fc_input_dim, *action_space_dim),
                nn.Softplus()
            )
        else:
            self.logits_nn = nn.Sequential(
                nn.Linear(self.fc_input_dim, action_space_dim),
            )

    def forward(self, input):
        main_body_out = self.main_body(input)
        if self.cont:
            mu = self.mu_nn(main_body_out)
            sigma = self.sigma_nn(main_body_out)
            sigma = torch.clamp(sigma, min=1e-3)
            return mu, sigma
        return self.logits_nn(main_body_out)

class Critic(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, conv):
        super(Critic, self).__init__()
        self.conv = conv

        if self.conv:
            self.fc_nn = nn.Sequential(
                nn.Conv2d(state_space_dim[0], 32, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.Flatten(), # (3136,)
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        else:
            self.fc_nn = nn.Sequential(
                nn.Linear(state_space_dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, input):
        return self.fc_nn(input)

class PPOAgent(Agent):
    def __init__(self, device, writer, actor_lr, critic_lr, gamma, conv,cont,
                 epsilon, epochs, tmax, entropy_weight, decay_steps=None, decay_rate=0.99, 
                 clip_grad_norm=None, save_path=None, load_path=None):
        self.device = device
        self.writer = writer
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.tmax = tmax
        self.conv = conv
        self.cont = cont
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.entropy_weight = entropy_weight
        self.clip_grad_norm = clip_grad_norm
        self.save_path = save_path
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

    def make_ppo_update(self):
        
        # unpack T timesteps
        s = torch.tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        a = torch.tensor(np.array(self.transitions["a"]), dtype=torch.float32 if self.cont else torch.int64).to(self.device) # (tmax,)
        r = torch.tensor(np.array(self.transitions["r"]), dtype=torch.float32).to(self.device) # (tmax)
        sprime = torch.tensor(np.array(self.transitions["sprime"]), dtype=torch.float32).to(self.device) # (tmax, state_space_dim)
        done = torch.tensor(np.array(self.transitions["done"]), dtype=torch.float32).to(self.device) # (tmax)
        old_log_probs = torch.as_tensor(np.array(self.transitions["log_probs"]), dtype=torch.float32).to(self.device) # (tmax,)
        masks = 1 - done # (tmax)

        if self.cont:
            mu, sigma = self.actor(s) # (tmax, action_space_dim)
            dist = torch.distributions.Normal(mu, sigma)
            chosen_log_probs = dist.log_prob(a).sum(-1) # (tmax,)
        else:
            logits = self.actor(s)
            log_probs = F.log_softmax(logits, dim=-1) # (tmax, num_action_choices,)
            chosen_log_probs = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1) # (tmax,)

        state_values = self.critic(s).squeeze(-1)
        last_state_value = self.critic(sprime[-1].unsqueeze(0)).squeeze(-1)

        R = last_state_value * masks[-1]
        returns = torch.zeros_like(r).to(self.device)
        for t in reversed(range(len(r))):
            R = r[t] + self.gamma * R
            returns[t] = R

        advantages = (returns - state_values).detach()

        ratios = torch.exp(chosen_log_probs - old_log_probs) # (tmax,)
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) # (tmax,)
        surrogate_obj = ratios * advantages # (tmax,)
        clipped_surrogate_obj = clipped_ratios * advantages # (tmax,)

        if self.cont:
            entropy_bonus = dist.entropy().mean()
        else:
            entropy_bonus = torch.distributions.Categorical(logits=logits).entropy().mean()

        policy_loss = -torch.min(surrogate_obj.mean(), clipped_surrogate_obj.mean()) - (self.entropy_weight * entropy_bonus)
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

        if self.writer is not None:
            self.writer.add_scalar("policy_loss", policy_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", state_value_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("entropy", entropy_bonus.item(), len(self.reward_history) + self.time_step)

        
    def update(self, s, sprime, a, r, done):

        # s is (num_envs, state_space_dim)
        # a is (num_envs, action_space_dim)
        # r is (num_envs,)
        # sprime is (num_envs, state_space_dim)
        # done is (num_envs,)

        self.transitions["s"].append(s[0])
        self.transitions["a"].append(a[0])
        self.transitions["r"].append(r[0])
        self.transitions["sprime"].append(sprime[0])
        self.transitions["done"].append(done[0])

        with torch.no_grad():
            if self.cont:
                mu, sigma = self.actor(self.process_state(s[0]).unsqueeze(0))
                dist = torch.distributions.Normal(mu.squeeze(0), sigma.squeeze(0))
                self.transitions["log_probs"].append(dist.log_prob(self.process_state(a[0])).sum(-1).cpu().detach())
            else:
                log_probs = self.actor(self.process_state(s[0]).unsqueeze(0))
                log_probs = log_probs[0, a[0]] # (1,)
                self.transitions["log_probs"].append(log_probs.cpu().detach()) 

        self.time_step += 1
        self.current_episode_rewards += r

        if self.time_step % self.tmax == 0 or done[0]:
            for epoch in range(self.epochs):
                self.make_ppo_update()
            self.reset_transitions()

        for env_idx in range(self.num_envs):
            if done[env_idx]:
                # if terminal, save/log/reset rewards, save model
                self.reward_history.append(self.current_episode_rewards[env_idx])

                if self.writer is not None:
                    self.writer.add_scalar("mean_episode_reward", np.mean(self.reward_history[-100:]), len(self.reward_history))
                    self.writer.add_scalar("episode_reward", self.current_episode_rewards[env_idx], len(self.reward_history))
                self.current_episode_rewards[env_idx] = 0.0

                if self.save_path is not None:
                    torch.save({
                        "actor_nn": self.actor.state_dict(),
                        "actor_optimiser": self.actor_optimiser.state_dict(),
                        "critic_nn": self.critic.state_dict(),
                        "critic_optimiser": self.critic_optimiser.state_dict(),
                    }, self.save_path)
        
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