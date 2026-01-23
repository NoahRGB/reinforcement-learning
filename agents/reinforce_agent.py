import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)                                           

from agents.agent import Agent

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class ReinforceAgent(Agent):
    def __init__(self, alpha, gamma, time_limit=10000):
        self.alpha = alpha
        self.gamma = gamma
        self.time_limit = time_limit

    def run_policy(self, s, t):
        probs = self.theta[s, :]
        actions = [i for i in range(self.action_space_size)]
        probs = tf.nn.softmax(probs).numpy()
        return np.random.choice(actions, 1, p=probs)[0]

    def update(self, s, sprime, a, r, done):
        self.current_episode_rewards += r
        self.steps.append((s, sprime, a, r))
        self.time_step += 1
        return self.time_step >= self.time_limit

    def initialise(self, state_space_size, action_space_size, start_state, resume=False):
        self.steps = []
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.current_episode_rewards = 0
        self.time_step = 0
        if not resume:
            self.theta = tf.Variable(np.zeros((state_space_size, action_space_size), np.float32), tf.float32)
            self.reward_history = []

    def get_action_probs(self, steps):
        action_probs = [tf.nn.softmax(self.theta[s, :]) for (s, sprime, a, r) in steps]
        chosen_action_probs = []
        for i in range(len(steps)):
            (s, sprime, a, r) = steps[i]
            chosen_action_probs.append(action_probs[i][a])
        return tf.stack(chosen_action_probs)

    def get_derivative(self, steps, G):

        with tf.GradientTape() as tape:
            tape.watch(self.theta)
            action_probs = self.get_action_probs(steps)
            L = tf.reduce_sum(tf.math.log(action_probs * G))
        return tape.gradient(L, self.theta)

    def finish_episode(self):
        
        returns = []
        G = 0
        for t in range(len(self.steps)-1, -1, -1):
            s, sprime, a, r = self.steps[t]
            G = self.gamma * G + r
            returns.insert(0, G)
        self.theta.assign_add(self.alpha * self.get_derivative(self.steps, returns))


        self.steps = []
        self.reward_history.append(self.current_episode_rewards)
        self.current_episode_rewards = 0
