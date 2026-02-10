import time

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.learn import learn
from utils.utils import detect_torch_device, create_tensorboard_writer
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
from agents.approximate.dqn_agent import DQNAgent

device = detect_torch_device()
writer = create_tensorboard_writer(comment="")

# =============== env =================
env = GymEnvironment("LunarLander-v3", False, render_mode=None)
# env = GymEnvironment("Acrobot-v1", False, render_mode=None)
# env = GymEnvironment("CartPole-v1", False, render_mode=None)
# env = GymEnvironment("MountainCar-v0", False, render_mode=None)
# env = GymEnvironment("Taxi-v3", False, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", False, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", False, render_mode="human")
# env = MazeEnvironment()

# =============== agent =================
# agent = QSigmaOffPolicyNstepSarsaAgent(n=2, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyMonteCarloAgent(epsilon=0.5, gamma=1.0, every_visit=False, decay_rate=0.99)

agent = ReinforceBaselineAgent(device, writer, policy_lr=0.01, state_value_lr=0.01, gamma=0.99, normalise=False)
# agent = ReinforceAgent(device, writer, lr=0.00001, gamma=0.99, normalise=False)
# agent = DQNAgent(device, writer, lr=0.0008, replay_memory_size=100000, minibatch_size=64, epsilon=0.9, gamma=0.99, decay_rate=0.99)
# agent = SemigradientSarsaAgent(device, normalise=False, lr=0.001, epsilon=0.3, gamma=1.0, decay_rate=1.0)
# agent = OffPolicyNstepSarsaAgentTB(n=4, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyNstepSarsaAgentISR(n=2, alpha=1.0, epsilon=0.1, gamma=0.9, expected=True)
# agent = OnPolicyNstepSarsaAgent(n=4, alpha=1.0, epsilon=0.5, gamma=0.4, expected=True, decay_rate=0.99)
# agent = SarsaAgent(alpha=1.0, epsilon=0.1, gamma=0.9, expected=True, decay_rate=0.99)
# agent = QLearningAgent(alpha=1.0, epsilon=0.9, gamma=0.99, decay_rate=0.99)
# agent = OnPolicyMonteCarloAgent(epsilon=0.9, gamma=0.99, every_visit=False, decay_rate=0.99)

# =============== learning =================

start = time.perf_counter()

episode_count = 1000
learning_rewards = learn(episode_count, env, agent, quiet=False)

print(f"Finished in {round(time.perf_counter() - start, 2)} seconds")


# episodes_per_run = 500 
# runs = 1 
# minibatch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# avg_episode_rewards = {size: np.zeros(episodes_per_run, dtype=np.float64) for size in minibatch_sizes}
# for minibatch_size in minibatch_sizes:
#     agent = DQNAgent(device, writer, lr=0.0007, replay_memory_size=10000, minibatch_size=minibatch_size, epsilon=0.9, gamma=0.99, decay_rate=0.99)
#     for run in range(0, runs):
#         run_rewards = learn(episodes_per_run, env, agent, quiet=True)
#         avg_episode_rewards[minibatch_size] += run_rewards
#         print(f"Finished run {run+1} of minibatch size {minibatch_size}")
#     avg_episode_rewards[minibatch_size] /= runs
#
# for size in minibatch_sizes:
#     if size == 16 or size == 256:
#         plt.plot([x for x in range(episodes_per_run)], avg_episode_rewards[size], alpha=0.7, label=f"{size}")
# plt.legend()
# plt.show()


writer.close()
