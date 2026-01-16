from agent import Agent

import numpy as np
import matplotlib.pyplot as plt

import random

import environment

class OffPolicyMonteCarloAgent(Agent):
    def __init__(self, environment, epsilon, discount_factor):
        super().__init__(environment)
        self.title = f"Off-policy Monte carlo"

        self.qtable = np.zeros((environment.maze_height, environment.maze_width, len(environment.actions)), dtype=np.float64)
        self.weights = np.zeros((environment.maze_height, environment.maze_width, len(environment.actions)), dtype=np.float64)
        self.epsilon = epsilon 
        self.discount_factor = discount_factor
        self.finished_episodes = False

        self.completed_iterations = 0
        self.trajectory_length_history = []
        self.reset_iteration()

    def get_optimal_action(self, state):
        y, x = state 
        legal_moves = self.environment.get_legal(state)
        q_values = self.qtable[y, x, :] # Fetch from the q-table the 4 q-values for this current state (The 4 q-values correspond to actions North, South, West, East)
        legal_q_values = self.qtable[y, x, legal_moves]
        best_q_value = legal_q_values.max() # identify the best q_value
        # print(f"legal_moves {legal_moves}, legal_q_values: {legal_q_values}, best_q_value: {best_q_value}")
        best_q_indices = np.argwhere(q_values == best_q_value).flatten().tolist() # find all those occurences of the max q-value
        best_q_indices = [index for index in best_q_indices if index in legal_moves]
        return best_q_indices

    def run_target_policy(self, state):
        # target policy!
        # greedy on the max q values discovered by the behaviour policy so far
        best_actions = self.get_optimal_action(state)
        # print(best_actions)
        return best_actions[0] if len(best_actions) > 0 else None

    def run_policy(self, state):
        # behaviour policy!
        legal_moves = self.environment.get_legal(state)
        # return random.choice(legal_moves)
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        return self.run_target_policy(state)

    def final_episode(self):
        saved_epsilon = self.epsilon
        self.learn(1, quiet=True)
        self.epsilon = saved_epsilon 
        self.finished_episodes = True

    def reset_iteration(self):
        # configuation for the current iteration
        self.time_step = 0
        self.state = self.environment.start_state
        self.target_state = self.environment.start_state
        self.done = False
        self.behaviour_done = False
        self.target_done = False
        self.current_iteration_path = []

        self.episodes = []
        self.visited = set()

    def iteration_step(self):
        self.time_step += 1

        # run target policy, save for displaying
        if not self.target_done:
            target_action = self.run_target_policy(self.target_state)
            if target_action != None:
                new_target_state, _, self.target_done = self.environment.step(target_action, self.target_state)
                self.target_state = new_target_state
                self.current_iteration_path.append((*new_target_state, (0, 0, 255)))
        else:
            self.trajectory_length_history.append(self.time_step)

        # run behaviour policy, record episode
        action = self.run_policy(self.state)
        new_state, reward, self.done = self.environment.step(action, self.state)
        self.episodes.append((self.state, action, reward))
        self.state = new_state
        # self.current_iteration_path.append((*self.state, (200, 200, 0)))

        if self.done:
            target_actions = {}
            for (state, _, _) in self.episodes:
                target_actions[tuple(state)] = self.run_target_policy(state)

            self.trajectory_length_history.append(self.time_step)
            self.completed_iterations += 1

            cumulative_reward = 0
            weight = 1
            for t in range(len(self.episodes) - 1, -1, -1):
                (y, x), action, reward = self.episodes[t]
                cumulative_reward = self.discount_factor * cumulative_reward + reward 
                self.weights[y, x, action] += weight 
                if self.weights[y, x, action] > 0:
                    self.qtable[y, x, action] += (weight / self.weights[y, x, action]) * (cumulative_reward - self.qtable[y, x, action])
                
                if action != target_actions[(y, x)]:
                    break

                # print("UPDATING")
                # weight = weight * ( 1 / ( 1 / len(self.environment.get_legal((y, x)) ) ) ) 
                weight /= (1.0 / len(self.environment.get_legal((y, x))))
            self.print_q_table(self.qtable)
            print()
            self.print_q_table(self.weights)

    def learn(self, iterations, quiet=False):
        for episode in range(iterations):
            self.reset_iteration()
            while not self.done:
                self.iteration_step()
            if not quiet: print(f"iteration {episode}, length: {self.trajectory_length_history[-1]}")

    def print_q_table(self, table):
        output = "\n"

        for i in range(0, self.environment.maze_height):
            row = ""
            for j in range(0, self.environment.maze_width):
                cell = ""
                vals = table[i, j, :]
                for val in vals:
                    cell += f"{val}/"
                row += f"{cell}     "
            output += f"{row}\n"

        print(output)

    def plot(self):
        plt.plot(self.trajectory_length_history)
        plt.ylabel("Trajectory Length")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.grid()
        plt.show()

    def __str__(self):
        return f""
