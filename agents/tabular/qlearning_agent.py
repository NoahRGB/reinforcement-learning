from agents.agent import Agent
from environments.spaces import DiscreteSpace, EnvType

import numpy as np

class QLearningAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, decay_rate=1.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.eval = False 
        self.decay_rate = decay_rate

    def initialise(self, state_space, action_space, start_state, num_envs):
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.current_episode_rewards = 0
        self.time_step = 0
        self.num_envs = num_envs

        self.qtable = np.full((self.state_space_size, self.action_space_size), 0.0)
        self.reward_history = []

    def finish_episode(self, episode_num):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.epsilon *= self.decay_rate

    def run_policy(self, s, t):
        if np.random.random() >= self.epsilon:
            best_actions = np.where(self.qtable[s[0], :] == self.qtable[s[0], :].max())
            return np.random.choice(best_actions[0])
        return np.random.choice([i for i in range(self.action_space_size)])

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r[0]
        update_target = r[0] + self.gamma * self.qtable[sprime[0], :].max()
        self.qtable[s[0], a] += self.alpha * (update_target - self.qtable[s[0], a])
        self.time_step += 1

    def toggle_eval(self):
        if not self.eval:
            self.epsilon_checkpoint = self.epsilon
            self.epsilon = 0.0
        else:
            self.epsilon = self.epsilon_checkpoint
        self.eval = not self.eval

    def get_supported_env_types(self):
        return [EnvType.SINGULAR]

    def get_supported_state_spaces(self):
        return [DiscreteSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
