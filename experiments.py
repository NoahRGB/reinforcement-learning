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

# env = AtariEnvironment("ALE/Pong-v5", NUM_ENVS, render_mode="human")
# env = AtariEnvironment("ALE/CrazyClimber-v5", NUM_ENVS, render_mode="human")
# env = GymEnvironment("CarRacing-v3", NUM_ENVS, render_mode=None, image_preprocess=True, continuous=False)
# env = GymEnvironment("Ant-v5", NUM_ENVS, render_mode=None)
# env = GymEnvironment("LunarLander-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Acrobot-v1", NUM_ENVS, render_mode=None)
# env = GymEnvironment("CartPole-v1", NUM_ENVS, render_mode=None)
# env = GymEnvironment("MountainCar-v0", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Taxi-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", NUM_ENVS, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", NUM_ENVS, render_mode=None)
# env = MazeEnvironment()

# =============== approximate agents =================

# agent = DQNAgent(device, writer, lr=0.001, conv=True,
#                          replay_memory_size=1000, replay_warmup_length=1000,
#                          C=1000, minibatch_size=32, gamma=0.99,
#                          epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1000,
#                          clip_grad_norm=None, update_freq=4,
#                          load_nn_path=None, save_nn_path=None)

# agent = PPOSingleAgent(device, writer, lr=0.001, gamma=0.99, conv=True, tmax=12, epsilon=0.5, epochs=3,
#                          entropy_weight=0.05, value_weight=1.0, clip_grad_norm=0.5, 
#                          save_path="results/temps/models/ppo_carracing.pt", load_path=None,)

# agent = PPOAgent(device, writer, lr=0.0001, gamma=0.99, conv=False, tmax=16, epsilon=0.4, epochs=4,
#                          entropy_weight=0.0, value_weight=1.0, clip_grad_norm=None, 
#                          save_path=None, load_path=None,)

# agent = A2CSingleAgent(device, writer, lr=0.0001, gamma=0.99, conv=True, tmax=16, decay_steps=None,
#                          entropy_weight=0.01, value_weight=0.5, clip_grad_norm=0.5,
#                          save_path=None, load_path=None,)

# agent = A2CAgent(device, writer, lr=0.001, gamma=0.99, conv=False, tmax=16,
#                          entropy_weight=0.05, value_weight=1.0, clip_grad_norm=0.5, 
#                          save_path=None, load_path=None,)

# agent = TDLambdaAgent(lambd=0.8, alpha=0.0001, epsilon=1.0, gamma=0.99, decay_rate=0.9) # not working

# agent = ReinforceAgent(device, writer, use_baseline=True, 
#                                policy_lr=0.001, state_value_lr=0.01, gamma=0.99,
#                                save_path="torch_models/reinforce_lunarlander_checkpoint.pt", load_path=None)

# agent = SemigradientSarsaAgent(device, writer, lr=0.001, 
#                                epsilon=0.01, gamma=0.99, decay_rate=0.0,
#                                load_path=None, save_path=None)



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

learning_rewards = learn(episode_count, env, agent, quiet=False)

finished = time.perf_counter()
print(f"Finished in {round(finished - start, 2)} seconds")

# plt.plot(learning_rewards)
# plt.show()

if writer is not None: writer.close()
