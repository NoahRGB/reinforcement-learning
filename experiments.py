import matplotlib.pyplot as plt
import numpy as np
import pickle

from learn import learn
from environments.gym_environment import GymEnvironment
from environments.maze_environment import MazeEnvironment

from agents.tabular.qlearning_agent import QLearningAgent
from agents.tabular.onpolicy_montecarlo_agent import OnPolicyMonteCarloAgent
from agents.tabular.offpolicy_montecarlo_agent import OffPolicyMonteCarloAgent
from agents.tabular.sarsa_agent import SarsaAgent
from agents.tabular.onpolicy_nstep_sarsa_agent import OnPolicyNstepSarsaAgent
from agents.tabular.offpolicy_nstep_sarsa_agent_isr import OffPolicyNstepSarsaAgentISR
from agents.tabular.offpolicy_nstep_sarsa_agent_tb import OffPolicyNstepSarsaAgentTB
from agents.tabular.qsigma_offpolicy_nstep_sarsa_agent import QSigmaOffPolicyNstepSarsaAgent

from agents.approximate.reinforce_agent import ReinforceAgent
from agents.approximate.reinforce_baseline_agent import ReinforceBaselineAgent 
from agents.approximate.semigradient_sarsa_agent import SemigradientSarsaAgent

# =============== env =================
# env = GymEnvironment("LunarLander-v3", False, render_mode=None)
# env = GymEnvironment("Acrobot-v1", False, render_mode=None)
# env = GymEnvironment("CartPole-v1", False, render_mode=None)
env = GymEnvironment("MountainCar-v0", False, render_mode=None)
# env = GymEnvironment("Taxi-v3", False, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", False, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", False, render_mode="human")
# env = MazeEnvironment()

# =============== agent =================
agent = ReinforceBaselineAgent(policy_lr=0.00001, state_value_lr=0.01, gamma=1.0, normalise=False)
# agent = ReinforceAgent(lr=0.001, gamma=0.99, normalise=False)
# agent = QSigmaOffPolicyNstepSarsaAgent(n=2, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyMonteCarloAgent(epsilon=0.5, gamma=1.0, every_visit=False, decay_rate=0.99)

# agent = SemigradientSarsaAgent(normalise=False, lr=0.001, epsilon=0.3, gamma=1.0, decay_rate=1.0)
# agent = OffPolicyNstepSarsaAgentTB(n=4, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyNstepSarsaAgentISR(n=2, alpha=1.0, epsilon=0.1, gamma=0.9, expected=True)
# agent = OnPolicyNstepSarsaAgent(n=4, alpha=1.0, epsilon=0.5, gamma=0.4, expected=True, decay_rate=0.99)
# agent = SarsaAgent(alpha=1.0, epsilon=0.1, gamma=0.9, expected=True, decay_rate=0.99)
# agent = QLearningAgent(alpha=1.0, epsilon=0.9, gamma=0.99, decay_rate=0.99)
# agent = OnPolicyMonteCarloAgent(epsilon=0.9, gamma=0.99, every_visit=False, decay_rate=0.99)

# =============== learning =================
episode_count = 1000
learning_rewards = learn(episode_count, env, agent, quiet=False)

plt.plot(learning_rewards)
plt.show()


