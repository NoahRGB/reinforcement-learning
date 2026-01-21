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
from agents.offpolicy_nstep_sarsa_agent import OffPolicyNstepSarsaAgent

episode_count = 200

# "CliffWalking-v1"
# "Taxi-v3"
# "FrozenLake-v1", is_slippery=True
# env = GymEnvironment("CliffWalking-v1", render_mode=None)
env = MazeEnvironment()

agent = OffPolicyNstepSarsaAgent(n=1, alpha=1.0, epsilon=0.9, gamma=0.9, expected=False)
# agent = NstepSarsaAgent(n=1, alpha=1.0, epsilon=0.9, gamma=0.9, expected=True, decay_rate=0.99)
# agent = SarsaAgent(alpha=1.0, epsilon=0.9, gamma=0.9, expected=True, decay_rate=0.99)
# agent = QLearningAgent(alpha=0.3, epsilon=0.9, gamma=0.8, decay_rate=0.99)
# agent = MonteCarloAgent(epsilon=0.3, gamma=0.9, every_visit=False, decay_rstete=1.0)

learning_rewards = learn(episode_count, env, agent, quiet=False)
# eval_rewards = evaluate(episode_count, env, agent)

# for line in agent.qtable:
    # print(line)

plt.plot(learning_rewards*-1)
# plt.plot(eval_rewards*-1)
plt.show()
