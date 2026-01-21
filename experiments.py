import matplotlib.pyplot as plt
import numpy as np
import pickle

from learn import learn, evaluate, parallel_learn_evaluate
from environments.gym_environment import GymEnvironment
from environments.maze_environment import MazeEnvironment

from agents.qlearning_agent import QLearningAgent
from agents.montecarlo_agent import MonteCarloAgent
from agents.sarsa_agent import SarsaAgent
from agents.nstep_sarsa_agent import NstepSarsaAgent

episode_count = 200

# "CliffWalking-v1"
# "Taxi-v3"
# "FrozenLake-v1", is_slippery=True
env = GymEnvironment("CliffWalking-v1", render_mode=None)
# env = MazeEnvironment()

# agent = NstepSarsaAgent(n=3, alpha=1.0, epsilon=0.9, gamma=0.9, expected=True, decay_rate=0.99)
# agent = SarsaAgent(alpha=1.0, epsilon=0.9, gamma=0.9, expected=True, decay_rate=0.99)
# agent = QLearningAgent(alpha=0.3, epsilon=0.9, gamma=0.8, decay_rate=0.99)
# agent = MonteCarloAgent(epsilon=0.3, gamma=0.9, every_visit=False, decay_rstete=1.0)

# learning_rewards = learn(episode_count, env, agent, quiet=False)
# eval_rewards = evaluate(episode_count, env, agent)

# for line in agent.qtable:
    # print(line)

# plt.plot(learning_rewards*-1)
# plt.plot(eval_rewards*-1)
# plt.show()

num_episodes = 100
num_runs = 100
alphas = np.arange(0.1, 1.05, 0.05)
avg_rewards = np.zeros(len(alphas)) 
# for i, alpha in enumerate(alphas):
#     for run in range(num_runs):
#         # agent = QLearningAgent(alpha=alpha, epsilon=0.1, gamma=1.0, decay_rate=1.0)
#         agent = SarsaAgent(alpha=alpha, epsilon=0.1, gamma=1.0, expected=True, decay_rate=0.99)
#         reward_history = learn(num_episodes, env, agent)
#         avg_rewards[i] += np.mean(reward_history)
#         print(f"finished run {run+1} for alpha {alpha}")
#     avg_rewards[i] /= num_runs 

# with open("expectedsarsa_alphas.pkl", "wb") as file:
#     pickle.dump(avg_rewards, file)

with open("qlearning_alphas.pkl", "rb") as file:
    qlearning_avg = pickle.load(file)
with open("sarsa_alphas.pkl", "rb") as file:
    sarsa_avg = pickle.load(file)
with open("expectedsarsa_alphas.pkl", "rb") as file:
    expectedsarsa_avg = pickle.load(file)

plt.plot(alphas, qlearning_avg, linewidth=1, label="Q-learning")
plt.plot(alphas, sarsa_avg, linewidth=1, label="Sarsa(0)")
plt.plot(alphas, expectedsarsa_avg, linewidth=1, label="Expected Sarsa(0)")
plt.xlabel("Alpha")
plt.ylabel("Average return over 100 episodes")
plt.legend()
plt.show()

