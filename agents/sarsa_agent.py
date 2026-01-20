from agents.agent import Agent

import numpy as np


class SarsaAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, decay_rate=1.0, time_limit=10000):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.eval = False
        self.time_limit = time_limit

    def initialise(self, state_space_size, action_space_size, resume=False):
        if not self.eval:
            if not resume:
                self.state_space_size = state_space_size
                self.action_space_size = action_space_size
                self.qtable = np.full((state_space_size, action_space_size), 0.0)
                self.reward_history = []
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        if not self.eval:
            self.epsilon *= self.decay_rate

    def run_policy(self, s):
        if np.random.random() >= self.epsilon or self.eval:
            best_actions = np.where(self.qtable[s, :] == self.qtable[s, :].max())
            return np.random.choice(best_actions[0])
        return np.random.choice([i for i in range(self.action_space_size)])

    def update(self, s, sprime, a, r):
        self.current_episode_rewards += r
        if not self.eval:
            aprime = self.run_policy(sprime)
            update_target = r + self.gamma * self.qtable[sprime, aprime]
            self.qtable[s, a] += self.alpha * (update_target - self.qtable[s, a])
        self.time_step += 1
        return self.time_step >= self.time_limit

    def toggle_eval(self):
        self.eval = not self.eval
