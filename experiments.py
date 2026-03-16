import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.learn import learn
from utils.evaluate import evaluate
from utils.utils import detect_torch_device, create_tensorboard_writer
from environments.gym_environment import GymEnvironment
from environments.atari_environment import AtariEnvironment
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
from agents.approximate.conv_dqn_agent import ConvDQNAgent

device = detect_torch_device()
writer = create_tensorboard_writer(comment="-pong")

# =============== env =================
# env = AtariEnvironment("ALE/Pong-v5", render_mode="human")
# env = GymEnvironment("LunarLander-v3", False, render_mode=None)
# env = GymEnvironment("Acrobot-v1", False, render_mode=None)
# env = GymEnvironment("CartPole-v1", False, render_mode=None)
# env = GymEnvironment("MountainCar-v0", False, render_mode=None)
# env = GymEnvironment("Taxi-v3", False, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", False, is_slippery=True, render_mode=None)
env = GymEnvironment("CliffWalking-v1", False, render_mode=None)
# env = MazeEnvironment()

# =============== agent =================
# agent = QSigmaOffPolicyNstepSarsaAgent(n=2, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyMonteCarloAgent(epsilon=0.5, gamma=1.0, every_visit=False, decay_rate=0.99)

# agent = ConvDQNAgent(device, writer, lr=1e-4, 
#                   replay_memory_size=10000, replay_warmup_length=10000,
#                   minibatch_size=32, 
#                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=150000,
#                   C=1000, gamma=0.99,
                  # save_nn_path="./torch_models/pong/pong_checkpoint.pt")
                  # load_nn_path="./torch_models/pong/pong_checkpoint.pt")
                  # load_nn_path="./results/bundles/pong/pong1/pong_checkpoint.pt")

# agent = DQNAgent(device, writer, lr=0.0001, replay_memory_size=10000, C=1000,
#                  minibatch_size=32, epsilon=0.9, gamma=0.99, decay_rate=0.99)

lr=0.1
# agent = ReinforceBaselineAgent(device, writer, policy_lr=0.01, state_value_lr=0.01, gamma=0.99, normalise=False)
# agent = ReinforceAgent(device, writer, lr=0.00001, gamma=0.99, normalise=False)
# agent = SemigradientSarsaAgent(device, normalise=False, lr=0.001, epsilon=0.3, gamma=1.0, decay_rate=1.0)
# agent = OffPolicyNstepSarsaAgentTB(n=4, alpha=1.0, epsilon=0.1, gamma=0.9)
# agent = OffPolicyNstepSarsaAgentISR(n=2, alpha=1.0, epsilon=0.1, gamma=0.9, expected=True)
# agent3 = OnPolicyNstepSarsaAgent(n=4, alpha=lr, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
# agent1 = SarsaAgent(alpha=lr, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
# agent2 = QLearningAgent(alpha=lr, epsilon=0.9, gamma=0.99, decay_rate=0.99)
# agent = OnPolicyMonteCarloAgent(epsilon=0.9, gamma=0.99, every_visit=True, decay_rate=0.99)




# =============== learning =================

# evaluate(agent, env, resume=False)

# start = time.perf_counter()
#
# episode_count = 150 
# learning_rewards = learn(episode_count, env, agent, eval_period=0, quiet=False)
#
# print(f"Finished in {round(time.perf_counter() - start, 2)} seconds")

# runs = 100
# avg_lr_1 = np.zeros(episode_count)
# avg_lr_2 = np.zeros(episode_count)
# avg_lr_3 = np.zeros(episode_count)
#
# for i in range(runs):
#     avg_lr_1 += np.array(learn(episode_count, env, agent1, eval_period=0, quiet=True))
#     avg_lr_2 += np.array(learn(episode_count, env, agent2, eval_period=0, quiet=True))
#     avg_lr_3 += np.array(learn(episode_count, env, agent3, eval_period=0, quiet=True))
#     print(f"done {i+1} / {runs}")
#
# avg_lr_1 = avg_lr_1 / runs
# avg_lr_2 = avg_lr_2 / runs
# avg_lr_3 = avg_lr_3 / runs
#
# plt.plot(avg_lr_1, color="purple", alpha=0.7, label="TD(0) SARSA")
# plt.plot(avg_lr_2, color="grey", alpha=0.7, label="Q-learning")
# plt.plot(avg_lr_3, color="blue", alpha=0.7, label="n-step SARSA (n=4)")
# plt.legend()
# plt.ylim(-400, 0)
# plt.xlabel("Episodes")
# plt.ylabel("Total episode reward")
# plt.savefig("cliffwalking_comparison.png", dpi=500, bbox_inches="tight")
# plt.show()

episode_count = 150
ns = [1, 3, 5, 10, 15]
runs = 100
avg_r = np.zeros((len(ns), episode_count))
for i in range(runs):
    for j in range(len(ns)):
        print(f"finished n={ns[j]} run={i}")
        agent = OnPolicyNstepSarsaAgent(n=ns[j], alpha=0.01, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
        r = learn(episode_count, env, agent, eval_period=0, quiet=True) 
        avg_r[j] += r

for n in range(len(ns)):
    avg_r[n] /= runs
    plt.plot(avg_r[n])

plt.show()









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
