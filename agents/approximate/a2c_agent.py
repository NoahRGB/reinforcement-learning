from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CombinedNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, conv, cont):
        super(CombinedNN, self).__init__()
        self.conv = conv
        self.cont = cont

        if self.conv:
            self.conv_nn = nn.Sequential(
                nn.Conv2d(state_space_dim[0], 32, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.Flatten() # (3136,)
            )

            self.conv_fc_nn = nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU()
            )
        else:
            self.non_conv_fc_nn = nn.Sequential(
                nn.Linear(state_space_dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

        if self.cont:
            self.mu_nn = nn.Sequential(
                nn.Linear(64, *action_space_dim),
            )

            self.sigma_nn = nn.Sequential(
                nn.Linear(64, *action_space_dim),
                nn.Softplus()
            )
        else:
            self.policy_nn = nn.Sequential(
                nn.Linear(64, action_space_dim),
            )

        self.state_value_nn = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, input):
        if self.conv:
            new_input = input / 255.0
            conv_out = self.conv_nn(new_input)
            fc_out = self.conv_fc_nn(conv_out)
            if self.cont:
                mu = self.mu_nn(fc_out)
                sigma = self.sigma_nn(fc_out)
                return (mu, sigma), self.state_value_nn(fc_out)
            return self.policy_nn(fc_out), self.state_value_nn(fc_out)
        else:
            fc_out = self.non_conv_fc_nn(input)
            if self.cont:
                mu = self.mu_nn(fc_out)
                sigma = self.sigma_nn(fc_out)
                return (mu, sigma), self.state_value_nn(fc_out)
            return self.policy_nn(fc_out), self.state_value_nn(fc_out)

class A2CAgent(Agent):
    def __init__(self, device, writer, lr, gamma, lam, conv, cont,
                 tmax, entropy_weight, value_weight=1.0, decay_steps=None, decay_rate=0.99, 
                 clip_grad_norm=None, save_path=None, load_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.lam = lam
        self.gamma = gamma
        self.conv = conv
        self.cont = cont
        self.tmax = tmax
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.clip_grad_norm = clip_grad_norm
        self.save_path = save_path
        self.load_path = load_path

    def process_state(self, s):
        return torch.tensor(s).to(self.device)

    def run_policy(self, s, t):
        # s is (num_envs, state_space_dim)
        if self.cont:
            (mu, sigma), _ = self.combined_nn(self.process_state(s)) # (num_envs, num_action_choices), (num_envs, num_action_choices)
            dist = torch.distributions.Normal(mu, sigma)
        else:
            logits, _ = self.combined_nn(self.process_state(s)) # (num_envs, num_action_choices)
            dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().cpu().numpy()
    
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
        
        self.reset_transitions()
        
        self.reward_history = []
        self.current_episode_rewards = np.zeros(num_envs)

        self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size, self.conv, self.cont).to(self.device)
        # self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)
        self.combined_optimiser = optim.RMSprop(self.combined_nn.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.combined_optimiser, step_size=self.decay_steps, gamma=self.decay_rate)

        # load saved models
        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.combined_nn.load_state_dict(checkpoint["nn"])
            self.combined_optimiser.load_state_dict(checkpoint["optimiser"])

    def make_a2c_update(self):
        
        # unpack T timesteps
        s = torch.tensor(np.array(self.transitions["s"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        a = torch.tensor(np.array(self.transitions["a"]), dtype=torch.float32 if self.cont else torch.int64).to(self.device) # (tmax, num_envs,)
        r = torch.tensor(np.array(self.transitions["r"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        sprime = torch.tensor(np.array(self.transitions["sprime"]), dtype=torch.float32).to(self.device) # (tmax, num_envs, state_space_dim)
        done = torch.tensor(np.array(self.transitions["done"]), dtype=torch.float32).to(self.device) # (tmax, num_envs)
        masks = 1 - done # (tmax, num_envs)

        # gather logprobs and pluck out chosen actions
        if not self.cont:
            logits, state_values = self.combined_nn(s) # (tmax, num_envs, action_space_dim), (tmax, num_envs, 1)
            log_probs = F.log_softmax(logits, dim=-1) # (tmax, num_envs, action_space_dim)
            chosen_log_probs = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1) # (tmax, num_envs)
        else:
            (mu, sigma), state_values = self.combined_nn(s) # (tmax, num_envs, action_space_dim), (tmax, num_envs, 1)
            dist = torch.distributions.Normal(mu, sigma)
            chosen_log_probs = dist.log_prob(a).sum(-1) # (tmax, num_envs)
        
        state_values = state_values.squeeze(-1) # (tmax, num_envs)

        # calculate advantages/GAE over T timesteps
        advantages = torch.zeros_like(r).to(self.device) # (tmax, num_envs)
        GAE = torch.zeros(self.num_envs, dtype=torch.float32).to(self.device) # (num_envs,)
        for t in reversed(range(self.tmax - 1)):
            delta = r[t] + self.gamma * state_values[t+1] * masks[t] - state_values[t]
            GAE = delta + self.gamma * self.lam * masks[t] + GAE
            advantages[t] = GAE

        # calculate state value loss, policy loss and entropybonus
        state_value_loss = advantages.pow(2).mean()
        policy_loss = -(chosen_log_probs * advantages.detach()).mean()
        if not self.cont:
            entropy_bonus = torch.distributions.Categorical(logits=logits).entropy().mean()
        else:
            entropy_bonus = dist.entropy().mean()
        combined_loss = policy_loss - (self.entropy_weight * entropy_bonus ) + (self.value_weight * state_value_loss)

        # backprop + (optionally) clip
        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()

        if self.decay_steps is not None:
            # self.scheduler.step()
            if self.time_step % self.decay_steps == 0:
                self.entropy_weight *= self.decay_rate

        if self.writer is not None:
            self.writer.add_scalar("policy_loss", policy_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", state_value_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("combined_loss", combined_loss.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("entropy", entropy_bonus.item(), len(self.reward_history) + self.time_step)
        
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

        self.time_step += 1
        self.current_episode_rewards += r

        if self.time_step % self.tmax == 0:
            self.make_a2c_update()
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
                        "nn": self.combined_nn.state_dict(),
                        "optimiser": self.combined_optimiser.state_dict(),
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