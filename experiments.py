import matplotlib.pyplot as plt
import numpy as np
import pickle

from learn import learn
from environments.gym_environment import GymEnvironment
from environments.maze_environment import MazeEnvironment

from agents.qlearning_agent import QLearningAgent
from agents.onpolicy_montecarlo_agent import OnPolicyMonteCarloAgent
from agents.offpolicy_montecarlo_agent import OffPolicyMonteCarloAgent
from agents.sarsa_agent import SarsaAgent
from agents.onpolicy_nstep_sarsa_agent import OnPolicyNstepSarsaAgent
from agents.offpolicy_nstep_sarsa_agent_isr import OffPolicyNstepSarsaAgentISR
from agents.offpolicy_nstep_sarsa_agent_tb import OffPolicyNstepSarsaAgentTB
from agents.qsigma_offpolicy_nstep_sarsa_agent import QSigmaOffPolicyNstepSarsaAgent
from agents.reinforce_agent import ReinforceAgent

episode_count = 100

# "CliffWalking-v1"
# "Taxi-v3"
# "FrozenLake-v1", is_slippery=True
# env = GymEnvironment("CliffWalking-v1", render_mode=None)
env = MazeEnvironment()


agent = ReinforceAgent(alpha=0.1, gamma=0.99)
# agent = QSigmaOffPolicyNstepSarsaAgent(n=2, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyMonteCarloAgent(epsilon=0.9, gamma=1.0, every_visit=False, decay_rate=1.0)
# agent = OffPolicyNstepSarsaAgentTB(n=4, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyNstepSarsaAgentISR(n=2, alpha=1.0, epsilon=0.1, gamma=0.9, expected=True)
# agent = OnPolicyNstepSarsaAgent(n=2, alpha=0.5, epsilon=0.1, gamma=0.9, expected=False, decay_rate=0.99)
# agent = SarsaAgent(alpha=1.0, epsilon=0.1, gamma=0.9, expected=True, decay_rate=0.99)
# agent = QLearningAgent(alpha=0.3, epsilon=0.9, gamma=0.8, decay_rate=0.99)
# agent = OnPolicyMonteCarloAgent(epsilon=0.3, gamma=0.9, every_visit=False, decay_rate=1.0)

learning_rewards = learn(episode_count, env, agent, quiet=False)

plt.plot(learning_rewards)
plt.show()
