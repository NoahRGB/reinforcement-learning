import os, time, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *
from environments import *
from agents.tabular import *
from agents.approximate import *

device = detect_torch_device()
writer = create_tensorboard_writer(comment="")
print(f"using device {device}")

NUM_ENVS = 1

# =============== environments =================

# env = AtariEnvironment("ALE/Pong-v5", NUM_ENVS, render_mode=None)
# env = AtariEnvironment("ALE/SpaceInvaders-v5", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Ant-v5", NUM_ENVS, render_mode=None)
env = GymEnvironment("MountainCar-v0", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Acrobot-v1", NUM_ENVS, render_mode="human")
# env = GymEnvironment("CartPole-v1", NUM_ENVS, render_mode="human")
# env = GymEnvironment("MountainCar-v0", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Taxi-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", NUM_ENVS, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", NUM_ENVS, render_mode=None)
# env = MazeEnvironment()

# =============== approximate agents =================

agent = DQNAgent(device, writer, lr=0.001, conv=False,
                         replay_memory_size=10000, replay_warmup_length=0,
                         C=1000, minibatch_size=32, gamma=0.99,
                         epsilon_start=0.99, epsilon_end=0.00, epsilon_decay_steps=100000,
                         clip_grad_norm=1.0, update_freq=4,
                         save_nn_path=None, load_nn_path=None)


# agent = ConvA2CAgent(device, writer, lr=0.001, gamma=0.99, tmax=5, entropy_weight=0.01, clip_grad_norm=0.5)
# agent = A2CAgent(device, writer, lr=0.001, gamma=0.99, 
#                  tmax=4, entropy_weight=0.05, clip_grad_norm=10.0, value_weight=1.0,)
#                  save_nn_path="./torch_models/a2c_cartpole_checkpoint.pt",)
                #  load_nn_path="./torch_models/a2c_bipedalwalker_checkpoint.pt",)


# agent = TDLambdaAgent(lambd=0.8, alpha=0.0001, epsilon=1.0, gamma=0.99, decay_rate=0.9) # not working

# agent = ReinforceBaselineAgent(device, writer, policy_lr=0.001, state_value_lr=0.01, gamma=0.99, normalise=True)
# agent = ReinforceAgent(device, writer, lr=0.001, gamma=0.99, normalise=False)
# agent = SemigradientSarsaAgent(device, writer, normalise=False, lr=0.001, epsilon=0.9, gamma=0.99, decay_rate=0.99)



# =============== tabular agents =================

# agent = OffPolicyNstepSarsaAgentTB(n=4, alpha=0.1, epsilon=0.1, gamma=0.99)
# agent = OffPolicyNstepSarsaAgentISR(n=4, alpha=1.0, epsilon=0.1, gamma=0.99, expected=False)
# agent = OnPolicyNstepSarsaAgent(n=4, alpha=1.0, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
# agent = SarsaAgent(alpha=1.0, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
# agent = QLearningAgent(alpha=1.0, epsilon=0.7, gamma=0.99, decay_rate=0.99)
# agent = OnPolicyMonteCarloAgent(epsilon=0.7, gamma=0.99, every_visit=False, decay_rate=0.999)



# =============== evaluating =================

# evaluate(agent, env, resume=False)

# =============== learning ==================


start = time.perf_counter()

episode_count = 500000

learning_rewards = learn(episode_count, env, agent, eval_period=0, quiet=False)

finished = time.perf_counter()
print(f"Finished in {round(finished - start, 2)} seconds")

# plt.plot(learning_rewards)
# plt.show()

if writer is not None: writer.close()
