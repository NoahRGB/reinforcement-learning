from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CombinedNN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, continuous_actions, action_space_mins=None, action_space_maxs=None):
        super(CombinedNN, self).__init__()
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim[0] if type(action_space_dim) == tuple else action_space_dim
        self.continuous_actions = continuous_actions
        self.action_space_mins = action_space_mins
        self.action_space_maxs = action_space_maxs

        self.fc_nn = nn.Sequential(
            nn.Linear(*state_space_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mu_nn = nn.Sequential(
            nn.Linear(64, self.action_space_dim),
        )

        self.sigma_nn = nn.Sequential(
            nn.Linear(64, self.action_space_dim),
            nn.Softplus(),
        )

        self.discrete_policy_nn = nn.Sequential(
            nn.Linear(64, self.action_space_dim),
            nn.LogSoftmax(dim=1)
        )

        self.state_value_nn = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, input):
        out = self.fc_nn(input)
        if self.continuous_actions:
            mus = self.mu_nn(out) # (batch, actions_space_dim,)
            sigmas = self.sigma_nn(out) + 1e-8 # (batch, actions_space_dim,)
            action_distributions = torch.stack((mus, sigmas), dim=-1) # (batch, actions_space_dim, 2)

            return action_distributions, self.state_value_nn(out)
        else:
            return self.discrete_policy_nn(out), self.state_value_nn(out)

class A2CAgent(Agent):
    def __init__(self, device, writer, lr, gamma, tmax, entropy_weight=0.01, clip_grad_norm=0.1, value_weight=1.0,
                 save_nn_path=None, load_nn_path=None):
        self.device = device
        self.writer = writer
        self.lr = lr
        self.entropy_weight = entropy_weight
        self.clip_grad_norm = clip_grad_norm
        self.tmax = tmax
        self.eval = False
        self.gamma = gamma
        self.save_nn_path = save_nn_path
        self.load_nn_path = load_nn_path
        self.value_weight = value_weight

    def process_state(self, s):
        return torch.tensor(s).float().to(self.device)

    def run_policy(self, s, t):
        if self.continuous_actions:
            with torch.no_grad():
                action_distribution, _ = self.combined_nn(self.process_state(s)) # (num_envs, action_space_dim, 2)
                return self.distribution_to_actions(action_distribution).cpu().numpy()
        else:
            with torch.no_grad():
                probs, _ = self.combined_nn(self.process_state(s)) # (num_envs, 6)
                probs = probs.cpu().numpy() # (num_envs, 6)
            probs = np.exp(probs) # (num_envs, 6)
            actions = np.array([np.random.choice(self.action_space_size, p=probs[i]) for i in range(probs.shape[0])]) # (num_envs,)
            return actions
    
    def distribution_to_actions(self, action_distributions):
        actions = torch.zeros((self.num_envs, *self.action_space_size)).to(self.device)
        for env in range(self.num_envs):
            env_action_distributions = action_distributions[env] # (action_space_dim, 2)
            all_env_mus = env_action_distributions[:, 0] # (action_space_dim,)
            all_env_sigmas = env_action_distributions[:, 1] # (action_space_dim)
            dist = torch.distributions.Normal(all_env_mus, all_env_sigmas)
            env_actions = dist.sample() # (action_space_dim,)
            env_actions = torch.clip(
                env_actions, 
                torch.as_tensor(self.action_space_mins).to(self.device), 
                torch.as_tensor(self.action_space_maxs).to(self.device)
            )
            actions[env] = env_actions
        return actions
    
    def reset_transitions(self):
        self.transitions = {
            "s":[],
            "a":[],
            "r":[],
            "sprime":[],
            "done":[]
        }

    def initialise(self, state_space, action_space, start_state, num_envs):
        
        self.reset_transitions()

        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
 
        self.num_envs = num_envs

        self.action_space_mins = None
        self.action_space_maxs = None
        self.continuous_actions = False
        if type(action_space) == ContinuousSpace:
            self.continuous_actions = True
            self.action_space_mins = action_space.min_bounds
            self.action_space_maxs = action_space.max_bounds

        self.time_step = 0
        self.total_episodes_completed = 0
        self.reward_history = []
        self.current_episode_rewards = np.zeros((self.num_envs,))

        self.combined_nn = CombinedNN(self.state_space_size, self.action_space_size, self.continuous_actions, self.action_space_mins, self.action_space_maxs).to(self.device)
        # self.combined_optimiser = optim.Adam(self.combined_nn.parameters(), lr=self.lr)
        self.combined_optimiser = optim.RMSprop(self.combined_nn.parameters(), lr=self.lr)


        if self.load_nn_path != None:
            checkpoint = torch.load(self.load_nn_path, weights_only=False)
            self.combined_nn.load_state_dict(checkpoint["nn"])
            self.combined_optimiser.load_state_dict(checkpoint["optimiser"])

    def make_update(self):
        all_policy_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_state_value_loss_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar (loss)
        all_entropy_total = torch.tensor(0.0, dtype=torch.float32).to(self.device) # scalar

        # self.transitions["s"] = (t_max, num_envs, 4, 84, 84)
        # self.transitions["sprime"] = (t_max, num_envs, 4, 84, 84)
        # self.transitions["a"] = (t_max, num_envs,)
        # self.transitions["r"] = (t_max, num_envs,)
        # self.transitions["done"] = (t_max, num_envs,)

        with torch.no_grad():
            last_terms = self.transitions["done"][-1] # (num_envs,) 
            last_sprimes = self.transitions["sprime"][-1] # (num_envs, 4, 84, 84)
            _, last_state_bootstraps = self.combined_nn(self.process_state(last_sprimes)) # (num_envs, 6), (num_envs, 1)
            R = (last_state_bootstraps.cpu().squeeze(1) * (1 - last_terms)).squeeze().float().to(self.device) # (num_envs,)

        for t in reversed(range(self.tmax)):
            
            all_states_t = self.transitions["s"][t] # (num_envs, 4, 84, 84)
            all_actions_t = self.transitions["a"][t] # (num_envs,)
            all_rewards_t = torch.as_tensor(self.transitions["r"][t], dtype=torch.float32).to(self.device) # (num_envs,)
            all_dones_t = torch.as_tensor(self.transitions["done"][t], dtype=torch.float32).to(self.device) # (num_envs,)

            # (num_envs,) + scalar * (num_envs,) = (num_envs,)
            R = all_rewards_t + self.gamma * R * (1 - all_dones_t) # (num_envs,)
            # R = all_rewards_t + self.gamma * R

            all_policy_predictions, all_state_value_predictions = self.combined_nn(
                torch.tensor(all_states_t, dtype=torch.float32).to(self.device)
            ) # (num_envs, action_space_dim), (num_envs, 1)

            # (num_envs,) - (num_envs,) = (num_envs,)
            with torch.no_grad():
                all_advantages = R - all_state_value_predictions.squeeze(1) # (num_envs,)
                # all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8) # normalise advantages

            if not self.continuous_actions:            
                all_log_probs = all_policy_predictions[torch.arange(self.num_envs), all_actions_t] # (num_envs,)
            else:
                all_log_probs = torch.zeros((self.num_envs,)).to(self.device)
                for env in range(self.num_envs):
                    env_policy_predictions = all_policy_predictions[env] # (action_space_dim, 2)
                    env_actions = all_actions_t[env] # (action_space_dim,)
                    env_log_prob = 0.0
                    for i in range(self.action_space_size[0]):
                        mu, sigma = env_policy_predictions[i]
                        dist = torch.distributions.Normal(mu, sigma)
                        env_log_prob += dist.log_prob(torch.tensor(env_actions[i]))
                    all_log_probs[env] = env_log_prob

            all_policy_loss_total += -(all_advantages * all_log_probs).mean() # scalar
            # all_state_value_loss_total += F.mse_loss(all_state_value_predictions.squeeze(1), R) # scalar
            all_state_value_loss_total += F.smooth_l1_loss(all_state_value_predictions.squeeze(1), R) # scalar


            if self.continuous_actions:
                dist = torch.distributions.Normal(env_policy_predictions[:, 0],
                                    env_policy_predictions[:, 1])
                all_entropy_total += dist.entropy().sum(dim=0).mean()
            else:
                all_entropy_total += -(torch.exp(all_policy_predictions) * all_policy_predictions).sum(dim=1).mean() # scalar

        # average losses over tmax steps (reduces scale of loss)
        all_policy_loss_total /= self.tmax
        all_state_value_loss_total /= self.tmax

        all_entropy_mean = all_entropy_total / self.tmax
        all_entropy_bonus = self.entropy_weight * all_entropy_mean

        combined_loss = all_policy_loss_total + self.value_weight * all_state_value_loss_total - all_entropy_bonus # scalar (loss)

        if self.writer != None:
            self.writer.add_scalar("policy_loss", all_policy_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("state_value_loss", all_state_value_loss_total.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("mean_policy_entropy", all_entropy_mean.item(), len(self.reward_history) + self.time_step)
            self.writer.add_scalar("combined_loss", combined_loss.item(), len(self.reward_history) + self.time_step)

        self.combined_optimiser.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.combined_nn.parameters(), self.clip_grad_norm)
        self.combined_optimiser.step()

        self.reset_transitions()

    def update(self, s, sprime, a, r, done):

        # s = (num_envs, 4, 84, 84)
        # sprime = (num_envs, 4, 84, 84)
        # a = (num_envs,)
        # r = (num_envs,)
        # done = (num_envs,)
        self.transitions["s"].append(s)
        self.transitions["a"].append(a)
        self.transitions["r"].append(r)
        self.transitions["sprime"].append(sprime)
        self.transitions["done"].append(done)

        self.current_episode_rewards += r
        for i in range(self.num_envs):
            if done[i]:
                self.total_episodes_completed += 1
                self.reward_history.append(self.current_episode_rewards[i])
                if self.writer != None:
                    self.writer.add_scalar("mean_episode_reward", np.mean(self.reward_history[-100:]), self.total_episodes_completed)
                    self.writer.add_scalar("episode_reward", self.current_episode_rewards[i], self.total_episodes_completed)
                self.current_episode_rewards[i] = 0.0

                if self.save_nn_path != None:
                    torch.save({
                        "nn": self.combined_nn.state_dict(),
                        "optimiser": self.combined_optimiser.state_dict(),
                    }, self.save_nn_path)


        # every tmax steps (or if episode ends), make update
        if (self.time_step+1) % self.tmax == 0:
            self.make_update()

        self.time_step += 1

    def finish_episode(self, episode_num):
        # not used for agents compatible with vectorised environments
        pass

    def toggle_eval(self):
        self.eval = not self.eval

    def get_supported_env_types(self):
        return [EnvType.SINGULAR, EnvType.VECTORISED]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace, ContinuousSpace]