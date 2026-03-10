import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v1", render_mode=None)
episodes, epsilon, alpha, gamma = 100, 0.1, 0.2, 0.99

actions = [i for i in range(env.action_space.n)]
qtable = np.zeros((env.observation_space.n, len(actions)))
rewards = []
td_errors = [] 

def policy(state):
    # epsilon greedy policy
    if np.random.random() < epsilon:
        return np.random.choice(actions) 
    else:
        return np.argmax(qtable[state])

for episode in range(episodes):

    state, info = env.reset()
    action = policy(state)
    done = False
    cumulative_reward = 0
    cumulative_tderr = 0

    while not done:

        new_state, reward, terminated, truncated, info = env.step(action)
        new_action = policy(new_state)
        td_error = reward + gamma * qtable[new_state][new_action] - qtable[state][action]
        qtable[state][action] += alpha * td_error 
        state, action = new_state, new_action
        cumulative_reward += reward
        cumulative_tderr += td_error
        done = terminated or truncated

    rewards.append(cumulative_reward)
    td_errors.append(cumulative_tderr)
    print(f"completed episode {episode}")
    if episode == 49:
        epsilon = 0.1

plt.plot(rewards, color="blueviolet")
# plt.plot(td_errors, color="blueviolet")
plt.title(f"ε=0.1, α={alpha}, γ={gamma}")
# plt.title(f"ε=1.0 for first 50 episodes then ε=0.1 for last 50 episodes, α={alpha}, γ={gamma}")
plt.xlabel("Episode")
# plt.ylabel("Cumulative episode TD error")
plt.ylabel("Cumulative episode reward")
plt.savefig("td0_agent1", bbox_inches='tight', dpi=200)
plt.show()

env.close()
