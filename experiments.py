import os, time, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *
from environments import *
from agents.tabular import *
from agents.approximate import *

NUM_ENVS = 1
EPISODES = 50000
USE_TENSORBOARD_LOGS = True
USE_NORMAL_LOGS = False
TITLE = "nothing"

device = detect_torch_device(quiet=False)
logger = Logger(use_normal_logs=USE_NORMAL_LOGS, use_tensorboard_logs=USE_TENSORBOARD_LOGS, parent_dir=f"results/temps/{TITLE}")


# =============== environments =================

# env = AtariEnvironment("ALE/Bowling-v5", NUM_ENVS, render_mode="human")
# env = AtariEnvironment("ALE/Boxing-v5", NUM_ENVS, render_mode="human")
# env = GymEnvironment("CarRacing-v3", NUM_ENVS, render_mode=None, image_preprocess=True, continuous=True)
# env = GymEnvironment("HalfCheetah-v5", NUM_ENVS, render_mode="human")
# env = GymEnvironment("LunarLander-v3", NUM_ENVS, render_mode=None, continuous=True)
# env = GymEnvironment("BipedalWalker-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Pendulum-v1", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Acrobot-v1", NUM_ENVS, render_mode=None)
env = GymEnvironment("CartPole-v1", NUM_ENVS, render_mode=None)
# env = GymEnvironment("MountainCar-v0", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Taxi-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", NUM_ENVS, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", NUM_ENVS, render_mode=None)
# env = MazeEnvironment()

# =============== approximate agents =================

# agent = DQNAgent(device, logger, job_title=TITLE, lr=0.001, conv=False,
#                  replay_memory_size=1000, replay_warmup_length=0,
#                  C=1000, minibatch_size=32, gamma=0.99,
#                  epsilon_start=1.0, epsilon_end=0.00, epsilon_decay_steps=10000,
#                  clip_grad_norm=0.5, update_freq=4,
#                  save_nn=True, load_nn_path=None)

# agent = PPOAgent(device, logger, job_title=TITLE, actor_lr=0.0003, critic_lr=0.0005, gamma=0.99, lam=0.95,
#                conv=False, cont=True, tmax=250006, epsilon=0.2, epochs=4, minibatch_size=64, 
#                decay_steps=None, decay_rate=None, entropy_weight=0.001, clip_grad_norm=0.5,
#                save_nn=False, load_path=None,)

agent = A2CAgent(device, logger, job_title=TITLE, actor_lr=0.00001, critic_lr=0.00001, gamma=0.99, lam=0.96,
               conv=True, cont=False, tmax=64, decay_steps=None, decay_rate=None,
               entropy_weight=0.0, clip_grad_norm=0.1,
               save_nn=False, load_path=None)

# agent = ReinforceAgent(device, logger, job_title=TITLE, use_baseline=True, 
#                                policy_lr=0.001, state_value_lr=0.01, gamma=0.99,
#                                save_nn=True, load_path=None)

# agent = SemigradientSarsaAgent(device, logger, job_title=TITLE, lr=0.001, 
#                                epsilon=0.99, gamma=0.99, decay_rate=0.99,
#                                save_nn=True, load_path=None)



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

learning_rewards = learn(EPISODES, env, agent, quiet=False)

finished = time.perf_counter()
print(f"Finished in {round(finished - start, 2)} seconds")

plt.plot(learning_rewards)
plt.show()