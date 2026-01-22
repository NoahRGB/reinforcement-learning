from agents.agent import Agent
import math
import numpy as np


class QSigmaOffPolicyNstepSarsaAgent(Agent):
    def __init__(self, n, alpha, epsilon, gamma, decay_rate=1.0, time_limit=10000):
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.time_limit = time_limit

    def initialise(self, state_space_size, action_space_size, start_state, resume=False):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        if not resume:
            self.qtable = np.full((state_space_size, action_space_size), 0.0)
            self.reward_history = []
        self.start_state = start_state
        self.states = {0: start_state}
        self.actions = {0: self.generate_action(self.start_state)}
        self.rewards = {}
        self.sigmas = {}
        self.isrs = {} # importance sampling ratios
        self.termination_time = np.inf
        self.current_episode_rewards = 0
        self.time_step = 0

    def finish_episode(self):
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
        self.states = {0: self.start_state}
        self.actions = {0: self.generate_action(self.start_state)}
        self.rewards = {}
        self.termination_time = np.inf
        self.time_step = 0
        self.epsilon *= self.decay_rate

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

    def run_target_policy(self, state):
        return np.random.choice(self.get_best_actions(state))

    def nstep_update(self, time_to_update, t):
        state_to_update = self.states[time_to_update]
        action_to_update = self.actions[time_to_update]

        target = 0
        for k in range(min(t+1, self.termination_time), time_to_update, -1):
            if k == self.termination_time:
                target = self.rewards[self.termination_time]
            else:
                expected_value = 0
                legal_actions = self.get_all_actions()
                best_actions = self.get_best_actions(self.states[k])
                chosen_action_prob = ((1/len(best_actions)) if self.actions[k] in best_actions else 0) 
                for a in legal_actions:
                    prob = ((1/len(best_actions)) if a in best_actions else 0)
                    expected_value += prob * self.qtable[self.states[k], a]

                target = (
                        self.rewards[k] +
                        self.gamma * expected_value +
                        self.gamma * (self.sigmas[k]*self.isrs[k] + (1-self.sigmas[k])*chosen_action_prob) * (target - self.qtable[self.states[k], self.actions[k]])
                )

        self.qtable[state_to_update, action_to_update] += (
                self.alpha * (target - self.qtable[state_to_update, action_to_update])
        )

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r

        if self.time_step < self.termination_time:
            self.states[self.time_step+1] = sprime
            self.rewards[self.time_step+1] = r
            if done:
                self.termination_time = self.time_step + 1
            else:
                self.actions[self.time_step+1] = self.generate_action(sprime)
                self.sigmas[self.time_step+1] = 0 

                all_actions = self.get_all_actions()
                best_actions = self.get_best_actions(self.states[self.time_step])
                if self.actions[self.time_step+1] in best_actions:
                    behaviour_probability = ((self.epsilon / len(all_actions)) + ((1-self.epsilon) / len(best_actions)))
                    target_probability = 1 / len(best_actions)
                else:
                    behaviour_probability = self.epsilon / len(all_actions)
                    target_probability = 0
                self.isrs[self.time_step+1] = target_probability / behaviour_probability

        time_to_update = self.time_step - self.n + 1
        if time_to_update >= 0:
            self.nstep_update(time_to_update, self.time_step)

        if done:
            # do n-1 extra updates
            extra_time_step = self.time_step + 1
            time_to_update = extra_time_step - self.n + 1
            while time_to_update <= (self.termination_time-1):
                if time_to_update >= 0:
                    self.nstep_update(time_to_update, extra_time_step)
                extra_time_step += 1
                time_to_update = extra_time_step - self.n + 1

        self.time_step += 1
        return self.time_step >= self.time_limit


