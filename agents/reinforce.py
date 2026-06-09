import numpy as np
import torch

import agents
import envs
import utils

class StateValueNN(torch.nn.Module):
    def __init__(self, num_inputs):
        super(StateValueNN, self).__init__()

        self.body = torch.nn.Sequential(
            torch.nn.Linear(*num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, inp):
        return self.body(inp)

class PolicyNN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyNN, self).__init__()

        self.body = torch.nn.Sequential(
            torch.nn.Linear(*num_inputs, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_outputs),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, inp):
        return self.body(inp)


class REINFORCE(agents.Agent):

    def __init__(self, policy_lr: float, state_value_lr: float, gamma: float, use_baseline: bool):
        self.policy_lr = policy_lr
        self.state_value_lr = state_value_lr
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = torch.device("cpu")

    def _get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            log_probs = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=log_probs)
            return dist.sample()
    
    def _setup(self, env: envs.Environment):
        self.state_space_dim = utils.detect_space_size(env.get_single_state_space())
        self.action_space_dim = utils.detect_space_size(env.get_single_action_space())
        self.policy_net = PolicyNN(self.state_space_dim, self.action_space_dim).to(self.device)
        self.state_value_net = StateValueNN(self.state_space_dim).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.state_value_optim = torch.optim.Adam(self.state_value_net.parameters(), lr=self.state_value_lr)

    def _improve(self, states: list, actions: list, rewards: list, sprimes: list, dones: list):        
        s = torch.cat(states).to(self.device) # (episode_len, state_dim)
        a = torch.cat(actions).to(self.device) # (episode_len,)
        r = torch.cat(rewards).to(self.device) # (episode_len,)
        sprime = torch.cat(sprimes).to(self.device) # (episode_len, state_dim)
        done = torch.cat(dones).to(self.device) # (episode_len,)

        state_values = self.state_value_net(s)
        state_values = state_values.squeeze(-1)

        episode_len = len(r)

        returns = torch.zeros(episode_len, dtype=torch.float32).to(self.device)
        G = 0.0
        for t in range(episode_len-1, -1, -1):
            G = self.gamma * G + r[t]
            returns[t] = G

        log_probs = self.policy_net(s) # (episode_len, action_dim)
        chosen_log_probs = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)
  
        policy_loss = - ((returns - (state_values.detach() if self.use_baseline else 0)) * chosen_log_probs).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.state_value_optim.zero_grad()
        state_value_loss = torch.nn.functional.mse_loss(state_values, returns)
        state_value_loss.backward()
        self.state_value_optim.step()

        self.logger.gradient_step_complete(["policy_loss", "state_value_loss"], [policy_loss.item(), state_value_loss.item()])
        self.logger.network_update({"pol_nn":self.policy_net.state_dict(), "sv_nn":self.state_value_net.state_dict()})

    def learn(self, total_timesteps: int, env: envs.Environment, logger: utils.Logger, seed: int = None):
        assert env.get_num_envs() == 1
        assert utils.is_space_discrete(env.get_single_action_space())
        assert env.is_conv() == False

        self.logger = logger
        current_game_states = torch.from_numpy(env.get_start_states()).float().to(self.device)
        utils.seed(seed)

        self._setup(env)

        states, actions, rewards, sprimes, dones = [], [], [], [], []
        for t in range(total_timesteps):

            self.logger.timestep_complete()

            current_actions = self._get_actions(current_game_states)
            current_sprimes, current_rewards, current_isterms, current_istruncs, current_infos = env.step(current_actions.cpu().numpy())

            states.append(current_game_states)
            actions.append(current_actions)
            rewards.append(torch.from_numpy(current_rewards).float().to(self.device))
            sprimes.append(torch.from_numpy(current_sprimes).float().to(self.device))
            dones.append(torch.from_numpy(current_isterms | current_istruncs).float().to(self.device))

            current_game_states = sprimes[-1]

            if "episode" in current_infos:
                self._improve(states, actions, rewards, sprimes, dones)
                states, actions, rewards, sprimes, dones = [], [], [], [], []

                done_idxs = current_infos["_episode"]
                completed_rewards = current_infos["episode"]["r"][done_idxs]
                for reward in completed_rewards:
                    self.logger.episode_complete(reward)

            self.logger.training_done()

    def to(self, device: torch.device):
        self.device = device