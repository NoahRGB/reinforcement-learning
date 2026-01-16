from agent import Agent

import numpy as np
import matplotlib.pyplot as plt

import random

class NstepExpectedSarsaAgent(Agent):
    def __init__(self, environment, n, epsilon, discount_factor, step_size=1.0):
        super().__init__(environment)
        self.title = f"On policy n-step expected Sarsa agent (decaying ε-greedy)"

        self.qtable = np.full((environment.maze_height, environment.maze_width, len(environment.actions)), 0.0)

        self.n = n
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.step_size = step_size

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
        self.state = self.environment.start_state # used for displaying the maze
        self.time_step = 0 
        self.done = False
        self.current_iteration_path = []

        self.states = {0: self.environment.start_state} 
        self.actions = {} 
        self.rewards = {} 
        self.epsilon *= 0.99

    def iteration_step(self):
        # make current move and record s', a, r
        current_state = self.states[self.time_step]
        action = self.run_policy(current_state)
        new_state, reward, self.done = self.environment.step(action, current_state)
        self.actions[self.time_step] = action
        self.states[self.time_step+1] = new_state
        self.rewards[self.time_step+1] = reward

        # check if there have been enough time steps to make 'n' updates
        time_to_update = self.time_step - self.n
        if time_to_update >= 0:
            state_to_update_y, state_to_update_x = self.states[time_to_update]
            action_to_update = self.actions[time_to_update]

            # sum the discounted 'n' observed rewards
            target = 0
            for t in range(time_to_update+1, time_to_update+self.n+1):
                target += (self.discount_factor**(t - time_to_update - 1)) * self.rewards[t]

            final_state_y, final_state_x = self.states[time_to_update+self.n]
            legal_actions = self.environment.get_legal(self.states[time_to_update+self.n])
            best_actions = self.get_best_actions(self.states[time_to_update+self.n])
            expected_value = 0
            for new_action in legal_actions:
                if new_action in best_actions:
                    prob = ((self.epsilon / len(legal_actions)) + ((1-self.epsilon) / len(best_actions)))
                else:
                    prob = (self.epsilon / len(legal_actions))
                expected_value += prob * self.qtable[final_state_y, final_state_x, new_action]

            target += (self.discount_factor**self.n) * expected_value

            self.qtable[state_to_update_y, state_to_update_x, action_to_update] += (
                    self.step_size * (target - self.qtable[state_to_update_y, state_to_update_x, action_to_update])
            )

        self.time_step += 1

        # recorded for displaying the environment
        self.state = self.states[self.time_step-1] 
        self.current_iteration_path.append((*self.states[self.time_step-1], (200, 200, 0)))

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
        return f"n = {self.n}, ε = {round(self.epsilon, 2)}, γ = {self.discount_factor}, α = {self.step_size}"

