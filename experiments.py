import matplotlib.pyplot as plt
import numpy as np

from learn import learn, evaluate, parallel_learn_evaluate
from environments.gym_environment import GymEnvironment
from environments.maze_environment import MazeEnvironment

from agents.qlearning_agent import QLearningAgent
from agents.montecarlo_agent import MonteCarloAgent
from agents.sarsa_agent import SarsaAgent

episode_count = 1000 

# "CliffWalking-v1"
# "Taxi-v3"
# "FrozenLake-v1", is_slippery=True
env = GymEnvironment("FrozenLake-v1", is_slippery=True, render_mode=None)
# env = MazeEnvironment()

# agent = SarsaAgent(alpha=0.1, epsilon=0.8, gamma=0.8, decay_rate=1.0)
# agent = QLearningAgent(alpha=0.3, epsilon=0.9, gamma=0.8, decay_rate=0.99)
# agent = MonteCarloAgent(epsilon=0.3, gamma=0.9, every_visit=False, decay_rate=1.0)

learning_rewards = learn(episode_count, env, agent, resume=False, timeouts=False, quiet=True)
# eval_rewards = evaluate(episode_count, env, agent, timeout=False)

plt.plot(learning_rewards)
# plt.plot(eval_rewards)
plt.show()
