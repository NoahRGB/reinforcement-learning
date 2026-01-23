from agent import Agent

import numpy as np
import matplotlib.pyplot as plt

import random

class TreeBackupOffPolicyNstepSarsaAgent(Agent):
    def __init__(self, environment, n, epsilon, discount_factor, step_size=1.0):
        super().__init__(environment)
        self.title = f"Off policy n-step Sarsa agent (tree backup)"

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
        # behaviour policy (ε-greedy)
        legal_moves = self.environment.get_legal(state)
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        else:
            # greedy
            return np.random.choice(self.get_best_actions(state))

    def run_target_policy(self, state):
        return np.random.choice(self.get_best_actions(state))

    def reset_iteration(self):
        # configuation for the current iteration
        self.time_step = 0 
        self.done = False
        self.current_iteration_path = []

        self.states = {0: self.environment.start_state} 
        self.actions = {} 
        self.rewards = {} 

        self.target_policy_time_step = 0 
        self.target_policy_current_state = self.environment.start_state
        self.target_policy_done = False
        self.behaviour_policy_done = False

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

            target = 0
            # calculate the initial target by summing over all actions taken at this
            # time step (the root of the tree currently)
            expected_value = 0
            current_state_y, current_state_x = self.states[self.time_step]
            legal_actions = self.environment.get_legal(self.states[self.time_step])
            best_actions = self.get_best_actions(self.states[self.time_step])
            for action in legal_actions:
                prob = ((1/len(best_actions)) if action in best_actions else 0)
                expected_value += prob * self.qtable[current_state_y, current_state_x, action]
            target = self.rewards[self.time_step] + self.discount_factor * expected_value

            # travel down the tree n steps and weight the expected values over all non chosen actions
            # by the probability of the actual action that was chosen
            for t in range(self.time_step, time_to_update, -1):
                expected_value = 0
                legal_actions = self.environment.get_legal(self.states[t])
                best_actions = self.get_best_actions(self.states[t])
                chosen_action_prob = ((1/len(best_actions)) if self.actions[t] in best_actions else 0) 
                for action in legal_actions:
                    if action != self.actions[t]:
                        current_state_y, current_state_x = self.states[t]
                        prob = ((1/len(best_actions)) if action in best_actions else 0)
                        expected_value += prob * self.qtable[current_state_y, current_state_x, action] 
                target = self.rewards[t] + self.discount_factor * expected_value + self.discount_factor * chosen_action_prob * target
                
            self.qtable[state_to_update_y, state_to_update_x, action_to_update] += (
                    self.step_size * (target - self.qtable[state_to_update_y, state_to_update_x, action_to_update])
            )


        self.time_step += 1

        self.state = new_state
        self.current_iteration_path.append((*self.state, (200, 200, 0)))

        if not self.target_policy_done:
            self.target_policy_time_step += 1
            target_policy_new_state, _, self.target_policy_done = self.environment.step(self.run_target_policy(self.target_policy_current_state), self.target_policy_current_state) 
            self.target_policy_current_state = target_policy_new_state
            self.current_iteration_path.append((*self.target_policy_current_state, (0, 0, 200)))

        if self.done:
            self.termination_time = self.time_step + 1
            self.trajectory_length_history.append(self.target_policy_time_step)
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


