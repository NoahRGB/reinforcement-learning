from agents.agent import Agent

import numpy as np


class QLearningAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, decay=False):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay = decay

    def initialise(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.qtable = np.full((state_space_size, action_space_size), 0.0)
        self.reward_history = []
        self.current_episode_rewards = 0

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        if self.decay:
            self.epsilon *= 0.99

    def run_policy(self, s):
        if np.random.random() < self.epsilon:
            return np.random.choice([i for i in range(self.action_space_size)])
        else:
            best_actions = np.where(self.qtable[s, :] == self.qtable[s, :].max())
            return np.random.choice(best_actions[0])

    def update(self, s, sprime, a, r):
        self.current_episode_rewards += r
        update_target = r + self.gamma * self.qtable[sprime, :].max()
        self.qtable[s, a] += self.alpha * (update_target - self.qtable[s, a])
