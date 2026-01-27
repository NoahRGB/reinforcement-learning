from agents.agent import Agent
from environments.spaces import DiscreteSpace

import numpy as np


class OnPolicyNstepSarsaAgent(Agent):
    def __init__(self, n, alpha, epsilon, gamma, expected, decay_rate=1.0, time_limit=10000):
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.expected = expected
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def initialise(self, state_space, action_space, start_state, resume=False):
        self.start_state = start_state
        self.state_space_size = state_space.dimensions
        self.action_space_size = action_space.dimensions
        if not resume:
            self.qtable = np.full((self.state_space_size, self.action_space_size), 0.0)
            self.reward_history = []
        self.states = {0: start_state}
        self.actions = {0: self.generate_action(start_state)}
        self.rewards = {}
        self.termination_time = np.inf
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.epsilon *= self.decay_rate
        self.states = {0: self.start_state}
        self.actions = {0: self.generate_action(self.start_state)}
        self.rewards = {}
        self.termination_time = np.inf
        self.time_step = 0

    def get_all_actions(self):
        return [i for i in range(self.action_space_size)]

    def get_best_actions(self, s):
        return np.where(self.qtable[s, :] == self.qtable[s, :].max())[0]

    def run_policy(self, s, t):
        return self.actions[t]

    def generate_action(self, s):
        if np.random.random() >= self.epsilon:
            return np.random.choice(self.get_best_actions(s))
        return np.random.choice(self.get_all_actions())

    def nstep_update(self, time_to_update):
        state_to_update = self.states[time_to_update]
        action_to_update = self.actions[time_to_update]

        target = 0
        for t in range(time_to_update+1, min(time_to_update+self.n, self.termination_time)+1):
            target += (self.gamma**(t - time_to_update - 1)) * self.rewards[t]

        if time_to_update + self.n < self.termination_time:
            if self.expected:
                all_actions = self.get_all_actions()
                best_actions = self.get_best_actions(self.states[time_to_update+self.n])
                update_target = 0
                for aprime in all_actions:
                    if aprime in best_actions:
                        prob = ((self.epsilon / len(all_actions)) + ((1-self.epsilon) / len(best_actions)))
                    else:
                        prob = (self.epsilon / len(all_actions))
                    update_target += prob * self.qtable[self.states[time_to_update+self.n], aprime]
                target += (self.gamma**self.n) * update_target
            else:
                target += (self.gamma**self.n) * self.qtable[self.states[time_to_update+self.n], self.actions[time_to_update+self.n]]

        self.qtable[state_to_update, action_to_update] += (
                self.alpha * (target - self.qtable[state_to_update, action_to_update])
        )

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r

        if self.time_step < self.termination_time:
            self.states[self.time_step+1] = sprime
            self.rewards[self.time_step+1] = r
            self.actions[self.time_step+1] = self.generate_action(sprime)
            if done:
                self.termination_time = self.time_step + 1

        time_to_update = self.time_step - self.n + 1
        if time_to_update >= 0:
            self.nstep_update(time_to_update)

        if done:
            # do n-1 extra updates
            extra_time_step = self.time_step
            while time_to_update <= (self.termination_time-1):
                extra_time_step += 1 
                time_to_update = extra_time_step - self.n + 1
                self.nstep_update(time_to_update)

        self.time_step += 1
        return self.time_step >= self.time_limit

    def get_supported_state_spaces(self):
        return [DiscreteSpace]

    def get_supported_action_spaces(self):
        return [DiscreteSpace]
