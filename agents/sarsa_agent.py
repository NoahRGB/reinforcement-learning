from agents.agent import Agent

import numpy as np


class SarsaAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, expected, decay_rate=1.0, time_limit=10000):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.expected = expected
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def initialise(self, state_space_size, action_space_size, start_state, resume=False):
        self.start_state = start_state
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.action = self.generate_action(start_state)
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.qtable = np.full((state_space_size, action_space_size), 0.0)
            self.reward_history = []

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.action = self.generate_action(self.start_state)
        self.epsilon *= self.decay_rate

    def get_all_actions(self):
        return [i for i in range(self.action_space_size)]

    def get_best_actions(self, s):
        return np.where(self.qtable[s, :] == self.qtable[s, :].max())[0]

    def run_policy(self, s, t):
        return self.action

    def generate_action(self, s):
        if np.random.random() >= self.epsilon:
            return np.random.choice(self.get_best_actions(s))
        return np.random.choice(self.get_all_actions())

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r
        if self.expected:
            # expected sarsa
            all_actions = self.get_all_actions()
            best_actions = self.get_best_actions(sprime)
            update_target = 0
            for aprime in all_actions:
                if aprime in best_actions:
                    prob = ((self.epsilon / len(all_actions)) + ((1-self.epsilon) / len(best_actions)))
                else:
                    prob = (self.epsilon / len(all_actions))
                update_target += prob * self.qtable[sprime, aprime]
            self.qtable[s, a] += self.alpha * (r + self.gamma * update_target - self.qtable[s, a])
            self.action = self.generate_action(sprime)
        else:
            aprime = self.generate_action(sprime) 
            update_target = r + self.gamma * self.qtable[sprime, aprime]
            self.qtable[s, a] += self.alpha * (update_target - self.qtable[s, a])
            self.action = aprime

        self.time_step += 1
        return self.time_step >= self.time_limit
