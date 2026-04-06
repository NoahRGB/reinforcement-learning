from agents.agent import Agent
from environments.spaces import DiscreteSpace, EnvType

import numpy as np

class SarsaAgent(Agent):
    def __init__(self, alpha, epsilon, gamma, expected, decay_rate=1.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.eval = False
        self.gamma = gamma
        self.expected = expected
        self.decay_rate = decay_rate

    def initialise(self, state_space, action_space, start_state, num_envs, resume=False):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions 
        self.action_space_size = action_space.dimensions
        self.num_envs = num_envs
        if not resume:
            self.qtable = np.full((self.state_space_size, self.action_space_size), 0.0)
            self.reward_history = []
        self.action = self.generate_action(start_state)
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self, episode_num):
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
        self.current_episode_rewards += r[0]
        if self.expected:
            # expected sarsa
            all_actions = self.get_all_actions()
            best_actions = self.get_best_actions(sprime[0])
            update_target = 0
            for aprime in all_actions:
                if aprime in best_actions:
                    prob = ((self.epsilon / len(all_actions)) + ((1-self.epsilon) / len(best_actions)))
                else:
                    prob = (self.epsilon / len(all_actions))
                update_target += prob * self.qtable[sprime[0], aprime]
            self.qtable[s[0], a] += self.alpha * (r[0] + self.gamma * update_target - self.qtable[s[0], a])
            self.action = self.generate_action(sprime[0])
        else:
            aprime = self.generate_action(sprime[0]) 
            update_target = r[0] + self.gamma * self.qtable[sprime[0], aprime]
            self.qtable[s[0], a] += self.alpha * (update_target - self.qtable[s[0], a])
            self.action = aprime

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
