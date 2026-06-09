from agents.agent import Agent
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class NN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(NN, self).__init__()
    
        self.fc1 = nn.Linear(*state_space_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_space_dim)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output

class SemigradientSarsaAgent(Agent):
    def __init__(self, device, logger, lr, epsilon, gamma, decay_rate=1.0, 
                 save_nn=False, load_path=None, job_title="semigradient_sarsa"):
        self.device = device
        self.logger = logger
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.load_path = load_path
        self.save_nn = save_nn
        self.job_title = job_title
        
    def prepare_state(self, s):
        return torch.as_tensor(s, dtype=torch.float32).to(self.device)

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions 
        self.action_space_size = action_space.dimensions 
        self.num_envs = num_envs

        self.nn = NN(self.state_space_size, self.action_space_size).to(self.device)
        self.optimiser = optim.Adam(self.nn.parameters(), lr=self.lr)

        if self.load_path is not None:
            checkpoint = torch.load(self.load_path)
            self.nn.load_state_dict(checkpoint["nn"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])

        self.reward_history = []
        self.action = self.generate_action(start_state)
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self, episode_num):
        self.reward_history.append(self.current_episode_rewards)
        self.action = self.generate_action(self.start_state)
        self.epsilon *= self.decay_rate

        self.logger.log("episode_reward", self.current_episode_rewards, episode_num)
        self.logger.log("mean_episode_reward", np.mean(self.reward_history[-100:]), episode_num)
        self.logger.save_logs()
        self.current_episode_rewards = 0

    def get_all_actions(self):
        return [i for i in range(self.action_space_size)]

    def get_best_action(self, s):
        with torch.no_grad():
            qvals = self.nn.forward(self.prepare_state(s)).cpu().numpy()
        return qvals.argmax()

    def run_policy(self, s, t):
        return [self.action]

    def generate_action(self, s):
        if np.random.random() >= self.epsilon:
            return self.get_best_action(s)
        return np.random.choice(self.get_all_actions())

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r[0]

        aprime = self.generate_action(self.prepare_state(sprime[0])) 

        self.optimiser.zero_grad()

        qs = self.nn.forward(self.prepare_state(s[0]))
        qsa = qs[a]

        if done[0]:
            target = torch.tensor(r[0]).to(self.device)
        else:
            with torch.no_grad():
                target = r[0] + self.gamma * self.nn.forward(self.prepare_state(sprime[0]))[aprime]

        td_err = target - qsa

        loss = 0.5 * td_err.pow(2)
        loss.backward()
        self.optimiser.step()

        self.action = aprime

        if done[0] and self.save_nn:
            self.logger.save_torch({
                "nn": self.nn.state_dict(),
                "optimiser": self.optimiser.state_dict(),
            }, f"{self.job_title}_model")

        self.time_step += 1

    def get_supported_env_types(self):
        return [EnvType.SINGULAR]

    def get_supported_state_spaces(self):
        return [ContinuousSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace, ContinuousSpace]

    def get_dump(self):
        return f"""

        """