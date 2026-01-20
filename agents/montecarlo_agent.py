from agents.agent import Agent

import numpy as np

class MonteCarloAgent(Agent):
    def __init__(self, epsilon, gamma, decay_rate=1.0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

    def run_policy(self, s):
        if np.random.random() < self.epsilon:
            return np.random.choice([i for i in range(self.action_space_size)])
        else:
            best_actions = np.where(self.qtable[s, :] == self.qtable[s, :].max())
            return np.random.choice(best_actions[0])

    def update(self, s, sprime, a, r):
        self.episodes.append((s, sprime, a, r))
        self.current_episode_rewards += r

    def initialise(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.qtable = np.full((state_space_size, action_space_size), 0.0)
        self.returns = np.zeros((state_space_size, action_space_size))
        self.visit_count = np.zeros((state_space_size, action_space_size))
        self.episodes = []
        self.reward_history = []
        self.current_episode_rewards = 0

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        G = 0
        for s, sprime, a, r in self.episodes:
            G = self.gamma * G + r
            # every visit
            self.visit_count[s, a] += 1
            self.returns[s, a] += G
            self.qtable[s, a] = self.returns[s, a] / self.visit_count[s, a]

        self.episodes = []
        self.epsilon *= self.decay_rate
