from environment import MazeEnvironment
from maze_display import show_agents

from on_policy_monte_carlo_agent import OnPolicyMonteCarloAgent
from off_policy_monte_carlo_agent import OffPolicyMonteCarloAgent

from sarsa_agent import SarsaAgent
from nstep_sarsa_agent import NstepSarsaAgent
from importance_sampling_off_policy_nstep_sarsa_agent import ImportanceSamplingOffPolicyNstepSarsaAgent
from tree_backup_off_policy_nstep_sarsa_agent import TreeBackupOffPolicyNstepSarsaAgent
from qsigma_off_policy_nstep_agent import QSigmaOffPolicyNstepAgent

from expected_sarsa_agent import ExpectedSarsaAgent
from nstep_expected_sarsa_agent import NstepExpectedSarsaAgent

from qlearning_agent import QLearningAgent
from double_qlearning_agent import DoubleQLearningAgent

import matplotlib.pyplot as plt


agents = [
        # QSigmaOffPolicyNstepAgent(MazeEnvironment(), n=2, epsilon=0.2, discount_factor=0.99),
        # TreeBackupOffPolicyNstepSarsaAgent(MazeEnvironment(), n=2, epsilon=0.1, discount_factor=0.99),
        # ImportanceSamplingOffPolicyNstepSarsaAgent(MazeEnvironment(), n=2, epsilon=0.1, discount_factor=0.99),
        # NstepExpectedSarsaAgent(MazeEnvironment(), n=1, epsilon=0.2, discount_factor=0.99),
        # NstepSarsaAgent(MazeEnvironment(), n=1, epsilon=0.1, discount_factor=0.99),
        # DoubleQLearningAgent(MazeEnvironment(), epsilon=0.9, discount_factor=0.99),
        # QLearningAgent(MazeEnvironment(), epsilon=0.9, discount_factor=0.99),
        # SarsaAgent(MazeEnvironment(), epsilon=0.8, discount_factor=0.99),
        # ExpectedSarsaAgent(MazeEnvironment(), epsilon=0.9, discount_factor=0.99),
        OnPolicyMonteCarloAgent(MazeEnvironment(), epsilon=0.9, discount_factor=0.99, every_visit=False),
]

# show_agents(agents)

agents[0].learn(100, quiet=False)
agents[0].plot()




# env = MazeEnvironment()
# done = False
# state = env.start_state
# time = 0
# while not done:
#     print(state)
#     action = agents[0].run_target_policy(state)
#     state, reward, done = env.step(action, state)
#     time += 1
# print(time)





