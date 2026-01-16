import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 16/01/2025
# tabular Q learning agent for CliffWalking
# using epsilon-greedy where ties are broken randomly

class Agent:
    def __init__(self, alpha, epsilon, gamma):
        self.alpha = alpha 
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make("CliffWalking-v0")

    def __del__(self):
        self.env.close()
        
    def run_policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        best_actions = np.where(self.qtable[state, :] == self.qtable[state, :].max())
        return np.random.choice(best_actions[0]) 

    def update(self, s, sprime, a, r):
        update_target = r + self.gamma * self.qtable[sprime, :].max()
        self.qtable[s, a] += self.alpha * (update_target - self.qtable[s, a])

    def train(self, episodes, decay_rate=1.0):
        self.qtable = np.full((self.env.observation_space.n, self.env.action_space.n), 0.0)
        self.reward_log = np.zeros(episodes)
        current_epsilon = self.epsilon

        for episode_num in range(episodes):
        
            # initialise episode
            state, info = self.env.reset()
            is_terminated, is_truncated = False, False
            total_reward = 0
        
            # episode loop
            while not is_terminated and not is_truncated:
                action = self.run_policy(state, current_epsilon) 
        
                new_state, reward, is_terminated, is_truncated, info = self.env.step(action)
                self.update(state, new_state, action, reward)
        
                state = new_state
                total_reward += reward
        
            current_epsilon *= decay_rate
            self.reward_log[episode_num] = total_reward

if __name__ == "__main__":
    episodes = 100 
    agent = Agent(alpha=0.8, epsilon=0.1, gamma=0.99) 
    agent.train(episodes, 0.0)

    plt.plot([i for i in range(0, episodes)], agent.reward_log * -1)
    plt.show()
    
    best = np.max(agent.qtable, axis=1).reshape(4, 12)
    plt.figure(figsize=(10, 5))
    plt.imshow(best, aspect="auto")
    plt.colorbar()
    plt.show()

