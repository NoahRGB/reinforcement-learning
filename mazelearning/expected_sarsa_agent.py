from agent import Agent

import numpy as np
import matplotlib.pyplot as plt

import random

class ExpectedSarsaAgent(Agent):
    def __init__(self, environment, epsilon, discount_factor, step_size=1.0):
        super().__init__(environment)
        self.title = f"On policy expected sarsa agent (decaying ε-greedy)"

        self.qtable = np.full((environment.maze_height, environment.maze_width, len(environment.actions)), 0.0)

        self.epsilon = epsilon
        self.step_size = step_size
        self.discount_factor = discount_factor

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
            # greedy
            return np.random.choice(self.get_best_actions(state))

    def reset_iteration(self):
        # configuation for the current iteration
        self.state = self.environment.start_state
        self.time_step = 0
        self.done = False
        self.current_iteration_path = []

        self.epsilon *= 0.99

    def iteration_step(self):
        self.time_step += 1
        action = self.run_policy(self.state)
        new_state, reward, self.done = self.environment.step(action, self.state)

        current_state_y, current_state_x = self.state
        new_state_y, new_state_x = new_state

        legal_actions = self.environment.get_legal(new_state)
        best_actions = self.get_best_actions(new_state)
        target = 0
        for new_action in legal_actions:
            if new_action in best_actions:
                prob = ((self.epsilon / len(legal_actions)) + ((1-self.epsilon) / len(best_actions)))
            else:
                prob = (self.epsilon / len(legal_actions))
            target += prob * self.qtable[new_state_y, new_state_x, new_action]

        self.qtable[current_state_y, current_state_x, action] += (
                self.step_size * (reward + self.discount_factor * target - self.qtable[current_state_y, current_state_x, action])
        )

        self.state = new_state

        self.current_iteration_path.append((*self.state, (200, 200, 0)))

        if self.done:
            self.trajectory_length_history.append(self.time_step)
            self.completed_iterations += 1

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
        return f"ε = {round(self.epsilon, 2)}, γ = {self.discount_factor}, α = {self.step_size}"
