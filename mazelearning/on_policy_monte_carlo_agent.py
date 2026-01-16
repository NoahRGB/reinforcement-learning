from agent import Agent

import numpy as np
import matplotlib.pyplot as plt

import random

class OnPolicyMonteCarloAgent(Agent):
    def __init__(self, environment, epsilon, discount_factor, every_visit):
        super().__init__(environment)
        self.title = f"On-policy decarying ε-greedy Monte carlo agent ({'every visit' if every_visit else 'first visit'})"

        self.qtable = np.full((environment.maze_height, environment.maze_width, len(environment.actions)), -1.)
        self.visit_count = np.zeros((environment.maze_height, environment.maze_width, len(environment.actions)), dtype=np.float64)
        self.returns = np.zeros((environment.maze_height, environment.maze_width, len(environment.actions)), dtype=np.float64)

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.every_visit = every_visit

        self.completed_iterations = 0
        self.trajectory_length_history = []
        self.reset_iteration()

    def get_best_actions(self, state):
        y, x = state 
        legal_moves = self.environment.get_legal(state)
        q_values = self.qtable[y, x, :]
        legal_q_values = self.qtable[y, x, legal_moves]
        best_q_value = legal_q_values.max()
        best_q_indices = np.argwhere(q_values == best_q_value).flatten().tolist()
        best_q_indices = [index for index in best_q_indices if index in legal_moves]
        return best_q_indices

    def run_policy(self, state):
        legal_moves = self.environment.get_legal(state)
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        else:
            # greedy policy
            return np.random.choice(self.get_best_actions(state))

    def reset_iteration(self):
        # configuation for the current iteration
        self.state = self.environment.start_state
        self.time_step = 0
        self.done = False
        self.current_iteration_path = []

        self.episodes = []
        self.visited = set()
        self.epsilon *= 0.99

    def iteration_step(self):
        self.time_step += 1
        
        # keep track of s', a, r for when the episode ends
        action = self.run_policy(self.state)
        new_state, reward, self.done = self.environment.step(action, self.state)
        self.episodes.append((self.state, action, reward))

        self.state = new_state
        self.current_iteration_path.append((*self.state, (200, 200, 0)))

        if self.done:
            self.trajectory_length_history.append(self.time_step)
            self.completed_iterations += 1

            # episode has ended, so do all the learning in one go
            cumulative_reward = 0
            for t in range(len(self.episodes) - 1, -1, -1):
                (y, x), action, reward = self.episodes[t]
                cumulative_reward = self.discount_factor * cumulative_reward + reward 
                if (not self.every_visit and (y, x, action) not in self.visited) or self.every_visit: 
                    self.visited.add((y, x, action))
                    self.visit_count[y, x, action] += 1
                    self.returns[y, x, action] += cumulative_reward 
                    self.qtable[y, x, action] = float(self.returns[y, x, action]) / self.visit_count[y, x, action]  

    def learn(self, iterations, quiet=False):
        for episode in range(iterations):
            self.reset_iteration()
            while not self.done:
                self.iteration_step()
            if not quiet: print(f"iteration {episode}, length: {self.trajectory_length_history[-1]}")

    def plot(self):
        plt.plot(self.trajectory_length_history)
        plt.ylabel("Trajectory Length")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.grid()
        plt.show()

    def __str__(self):
        return f"ε = {round(self.epsilon, 2)}, γ = {self.discount_factor}"
