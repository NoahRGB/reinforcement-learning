from agents.agent import Agent
from environments.spaces import DiscreteSpace

import numpy as np

class OffPolicyMonteCarloAgent(Agent):
    def __init__(self, epsilon, gamma, every_visit=True, decay_rate=1.0, time_limit=10000):
        self.epsilon = epsilon
        self.gamma = gamma
        self.every_visit = every_visit
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def get_best_actions(self, s):
        return np.where(self.qtable[s, :] == self.qtable[s, :].max())[0]

    def run_policy(self, s, t):
        if np.random.random() >= self.epsilon:
            return np.random.choice(self.get_best_actions(s))
        return np.random.choice([i for i in range(self.action_space_size)])

    def run_target_policy(self, s):
        return self.get_best_actions(s)[0]

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r
        self.episodes.append((s, sprime, a, r))
        self.time_step += 1
        return self.time_step >= self.time_limit

    def initialise(self, state_space, action_space, start_state, resume=False):
        self.episodes = []
        self.visits = set()
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.qtable = np.full((self.state_space_size, self.action_space_size), 0.0)
            self.returns = np.zeros((self.state_space_size, self.action_space_size))
            self.visit_count = np.zeros((self.state_space_size, self.action_space_size))
            self.c = np.zeros((self.state_space_size, self.action_space_size))
            self.reward_history = []

    def finish_episode(self):
        G = 0
        W = 1
        for t in range(len(self.episodes)-1, -1, -1):
            s, sprime, a, r = self.episodes[t]
            G = self.gamma * G + r
            self.visits.add((s, a))
            self.visit_count[s, a] += 1
            self.c[s, a] += W
            self.returns[s, a] += G
            self.qtable[s, a] += (W / self.c[s, a]) * (G - self.qtable[s, a]) 
            if a != self.run_target_policy(s):
                break
            all_actions = self.get_all_actions()
            best_actions = self.get_best_actions(s)
            if a in best_actions:
                prob = ((self.epsilon / len(all_actions)) + ((1-self.epsilon) / len(best_actions)))
            else:
                prob = (self.epsilon / len(all_actions))
            W *= 1 / (prob)

        self.episodes = []
        self.visits = set()
        self.epsilon *= self.decay_rate

        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0

    def get_supported_state_spaces(self):
        return [DiscreteSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
