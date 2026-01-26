from agents.agent import Agent

import numpy as np


class QLearningAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, decay_rate=1.0, time_limit=10000):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def initialise(self, state_space_size, action_space_size, start_state, resume=False):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.qtable = np.full((state_space_size, action_space_size), 0.0)
            self.reward_history = []

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.epsilon *= self.decay_rate

    def run_policy(self, s, t):
        if np.random.random() >= self.epsilon:
            best_actions = np.where(self.qtable[s, :] == self.qtable[s, :].max())
            return np.random.choice(best_actions[0])
        return np.random.choice([i for i in range(self.action_space_size)])

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r
        update_target = r + self.gamma * self.qtable[sprime, :].max()
        self.qtable[s, a] += self.alpha * (update_target - self.qtable[s, a])
        self.time_step += 1
        return self.time_step >= self.time_limit
