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

import numpy as np

env = MazeEnvironment()

agents = [
        # QSigmaOffPolicyNstepAgent(MazeEnvironment(), n=2, epsilon=0.2, discount_factor=0.99),
        # TreeBackupOffPolicyNstepSarsaAgent(MazeEnvironment(), n=2, epsilon=0.1, discount_factor=0.99),
        # ImportanceSamplingOffPolicyNstepSarsaAgent(MazeEnvironment(), n=2, epsilon=0.1, discount_factor=0.99),
        # NstepExpectedSarsaAgent(MazeEnvironment(), n=1, epsilon=0.2, discount_factor=0.99),
        # NstepSarsaAgent(MazeEnvironment(), n=1, epsilon=1.0, discount_factor=0.99),
        # DoubleQLearningAgent(MazeEnvironment(), epsilon=0.9, discount_factor=0.99),
        QLearningAgent(env, epsilon=0.1, discount_factor=1.0),
        # SarsaAgent(MazeEnvironment(), epsilon=0.8, discount_factor=0.99),
        # ExpectedSarsaAgent(MazeEnvironment(), epsilon=0.9, discount_factor=0.99),
        OnPolicyMonteCarloAgent(env, epsilon=1.0, discount_factor=1.0, every_visit=True),
]

# show_agents(agents, env)


# epsilons = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# epsilon_totals = {}

# for epsilon in epsilons:
#     epsilon_totals[epsilon] = []
#     for trial in range(2):
#         # agent = QLearningAgent(env, epsilon=epsilon, discount_factor=1.0)
#         agent = OnPolicyMonteCarloAgent(env, epsilon=0.5, discount_factor=1.0, every_visit=True)

#         epsilon_totals[epsilon].append(agent.learn(1000, quiet=False))
#         print(f"epsilon {epsilon} trial {trial} complete")

# for epsilon, rewards in epsilon_totals.items():
#     plt.plot(np.array(rewards).mean(axis=0), label=f'ε = {epsilon}')

# plt.legend()
plt.show()


_ = agents[0].learn(1000, quiet=False)
_ = agents[1].learn(1000, quiet=False)

env.maze[12][19] = 1
env.maze[11][19] = 1
env.maze[10][19] = 1
env.maze[9][19] = 1
env.maze[8][19] = 1
env.maze[7][19] = 1

show_agents(agents, env)

# r2 = agents[0].learn(500, quiet=False)

# plt.plot(r2)
# plt.show()


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





