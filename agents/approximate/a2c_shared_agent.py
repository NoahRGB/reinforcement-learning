import pickle

from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CombinedNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, cont):
        super(CombinedNN, self).__init__()
        self.cont = cont

        self.body = nn.Sequential(
            nn.Linear(state_space_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        if self.cont:
            self.mu_head = nn.Linear(64, *action_space_dim)
            self.log_sigma_head = nn.Parameter(torch.zeros(*action_space_dim))
        else:
            self.actor_head = nn.Linear(64, action_space_dim)
            
        self.critic_head = nn.Linear(64, 1)

    def forward(self, inp):
        main_out = self.body(inp)
        critic_out = self.critic_head(main_out)
        if self.cont:
            mu = self.mu_head(main_out)
            log_sigma = self.log_sigma_head
            return (mu, log_sigma.exp()), critic_out
        else:
            actor_out = self.actor_head(main_out)
            return actor_out, critic_out

class SharedA2CAgent(Agent):
    def __init__(self, device, logger, lr, gamma, lam, conv, cont,
                 tmax, value_weight, entropy_weight, decay_steps=None, decay_rate=0.99, 
                 clip_grad_norm=None, save_nn=False, load_path=None, job_title="a2c"):
        self.device = device
        self.logger = logger
        self.job_title = job_title
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.tmax = tmax
        self.conv = conv
        self.cont = cont
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.clip_grad_norm = clip_grad_norm
        self.save_nn = save_nn
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s, dtype=torch.float32).to(self.device)

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim)
        if self.cont:
            (mu, sigma), _ = self.combined_nn(torch.tensor(s, dtype=torch.float32).to(self.device)) # (num_envs, action_space_dim,)
            dist = torch.distributions.Normal(mu, sigma)
            return dist.sample().cpu().numpy() # (num_envs, action_space_dim,)
        else:
            logits, _ = self.combined_nn(torch.tensor(s, dtype=torch.float32).to(self.device))
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample().cpu().numpy() # (num_envs, action_space_dim,)
    
    def reset_transitions(self):
        self.transitions = {
            "s": [],
            "a": [],
            "r": [],
            "sprime": [],
            "done": [],
        }

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs
        self.num_action_choices = action_space.num

        self.time_step = 0
        self.gradient_update_count = 0
        self.reward_record = -np.inf
        
        self.reset_transitions()
        
        self.reward_history = []
        self.current_episode_rewards = np.zeros(num_envs)

        self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size, self.cont).to(self.device)
        self.optimiser = optim.RMSprop(self.combined_nn.parameters(), lr=self.lr, eps=1e-05)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.combined_nn.load_state_dict(checkpoint["combined_nn"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])

    def make_a2c_update(self):

        self.gradient_update_count += 1
        
        # unpack T timesteps
        s = torch.tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        a = torch.tensor(np.array(self.transitions["a"]), dtype=torch.float32 if self.cont else torch.int64).to(self.device) # (tmax, num_envs)
        r = torch.tensor(np.array(self.transitions["r"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        sprime = torch.tensor(np.array(self.transitions["sprime"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        done = torch.tensor(np.array(self.transitions["done"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        masks = 1 - done # (tmax, num_envs)

        if self.cont:
            (mu, sigma), state_values = self.combined_nn(s) # (tmax, num_envs, action_space_dim), (tmax, num_envs, 1)
        else:
            logits, state_values = self.combined_nn(s) # (tmax, num_envs, num_actions), (tmax, num_envs, 1)
        state_values = state_values.squeeze(-1)

        _, last_state_values = self.combined_nn(sprime[-1].unsqueeze(0)) # (1, num_envs, 1)
        last_state_values = last_state_values.squeeze()

        G = last_state_values
        returns = torch.zeros_like(r, dtype=torch.float32).to(self.device)
        for t in reversed(range(len(r))):
            G = r[t] + self.gamma * masks[t] * G
            returns[t] = G

        advantages = (returns - state_values)

        if self.cont:
            dist = torch.distributions.Normal(mu, sigma)
            chosen_log_probs = dist.log_prob(a).sum(-1) # (tmax, num_envs)
            entropy_bonus = dist.entropy().mean()
        else:
            log_probs = F.log_softmax(logits, dim=-1) # (tmax, num_envs, num_action_choices,)
            chosen_log_probs = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1) # (tmax, num_envs)
            entropy_bonus = torch.distributions.Categorical(logits=logits).entropy().mean()

        policy_loss = -(chosen_log_probs * advantages).mean() - (self.entropy_weight * entropy_bonus)
        state_value_loss = F.mse_loss(state_values, returns)

        combined_loss = policy_loss + self.value_weight * state_value_loss
        self.optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.optimiser.step()

        if self.decay_steps is not None:
            # self.scheduler.step()
            if self.time_step % self.decay_steps == 0:
                self.entropy_weight *= self.decay_rate

        self.logger.log("combined_loss", combined_loss.item(), step=self.gradient_update_count)
        self.logger.log("policy_loss", policy_loss.item(), step=self.gradient_update_count)
        self.logger.log("state_value_loss", state_value_loss.item(), step=self.gradient_update_count)
        self.logger.log("entropy", entropy_bonus.item(), step=self.gradient_update_count)
        if self.cont:self.logger.log("sigma", sigma.mean().item(), step=self.gradient_update_count)

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

        self.time_step += self.num_envs
        self.current_episode_rewards += r

        if self.time_step % (self.tmax * self.num_envs) == 0:
            self.make_a2c_update()
            self.reset_transitions()
            self.logger.save_logs()

        for env_idx in range(self.num_envs):
            if done[env_idx]:
                # if terminal, save/log/reset rewards, save model
                self.reward_history.append(self.current_episode_rewards[env_idx])
                mean_recent_reward = np.mean(self.reward_history[-100:])
                self.logger.log("mean_episode_reward", mean_recent_reward, step=self.time_step)
                self.logger.log(f"episode_reward_{self.job_title}", self.current_episode_rewards[env_idx], step=self.time_step)
                self.current_episode_rewards[env_idx] = 0.0

                if self.save_nn and mean_recent_reward > self.reward_record:
                    self.reward_record = mean_recent_reward
                    self.logger.save_torch({
                        "combined_nn": self.combined_nn.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
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