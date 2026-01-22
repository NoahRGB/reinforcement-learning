from agents.agent import Agent

import numpy as np

class OnPolicyMonteCarloAgent(Agent):
    def __init__(self, epsilon, gamma, every_visit=True, decay_rate=1.0, time_limit=10000):
        self.epsilon = epsilon
        self.gamma = gamma
        self.every_visit = every_visit
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def run_policy(self, s, t):
        if np.random.random() >= self.epsilon:
            best_actions = np.where(self.qtable[s, :] == self.qtable[s, :].max())
            return np.random.choice(best_actions[0])
        return np.random.choice([i for i in range(self.action_space_size)])

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r
        self.episodes.append((s, sprime, a, r))
        self.time_step += 1
        return self.time_step >= self.time_limit

    def initialise(self, state_space_size, action_space_size, start_state, resume=False):
        self.episodes = []
        self.visits = set()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.qtable = np.full((state_space_size, action_space_size), 0.0)
            self.returns = np.zeros((state_space_size, action_space_size))
            self.visit_count = np.zeros((state_space_size, action_space_size))
            self.reward_history = []

    def finish_episode(self):
        G = 0
        for t in range(len(self.episodes)-1, -1, -1):
            s, sprime, a, r = self.episodes[t]
            G = self.gamma * G + r
            if self.every_visit or ((s, a) not in self.visits):
                self.visits.add((s, a))
                self.visit_count[s, a] += 1
                self.returns[s, a] += G
                self.qtable[s, a] = self.returns[s, a] / self.visit_count[s, a]

        self.episodes = []
        self.visits = set()
        self.epsilon *= self.decay_rate

        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
