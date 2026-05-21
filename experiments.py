import os, time, sys, random, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import matplotlib.pyplot as plt
import numpy as np
import pickle
import gymnasium as gym
import torch

from utils import *
from environments import *
from agents.tabular import *
from agents.approximate import *

NUM_ENVS = 1
EPISODES = 100000
SEED = None
USE_TENSORBOARD_LOGS = True
USE_NORMAL_LOGS = False
TITLE = "drqn_tests"

device = detect_torch_device(quiet=False)
logger = Logger(use_normal_logs=USE_NORMAL_LOGS, use_tensorboard_logs=USE_TENSORBOARD_LOGS, parent_dir=f"results/temps/{TITLE}")

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

# gym.register(id="custom/BallEnv-v0", entry_point=BallEnv, max_episode_steps=1)
gym.register(id="custom/RNNAntEnv-v0", entry_point=RNNAntEnv, max_episode_steps=30)

# =============== environments =================

env = AtariEnvironment("ALE/Pong-v5", NUM_ENVS, render_mode=None, seed=SEED)
# env = AtariEnvironment("ALE/Boxing-v5", NUM_ENVS, render_mode="human")
# env = GymEnvironment("CarRacing-v3", NUM_ENVS, render_mode=None, image_preprocess=True, continuous=True)
# env = GymEnvironment("Humanoid-v5", NUM_ENVS, render_mode=None)
# env = GymEnvironment("LunarLander-v3", NUM_ENVS, render_mode=None, continuous=True)
# env = GymEnvironment("BipedalWalker-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Pendulum-v1", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Acrobot-v1", NUM_ENVS, render_mode=None)
# env = GymEnvironment("CartPole-v1", NUM_ENVS, render_mode=None, seed=SEED)
# env = GymEnvironment("MountainCar-v0", NUM_ENVS, render_mode=None)
# env = GymEnvironment("Taxi-v3", NUM_ENVS, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", NUM_ENVS, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", NUM_ENVS, render_mode=None)
# env = MazeEnvironment()
# env = GymEnvironment("custom/BallEnv-v0", NUM_ENVS, render_mode="human")
# env = GymEnvironment("custom/RNNAntEnv-v0", NUM_ENVS, render_mode=None)

# =============== approximate agents =================

# agent = PrioritisedDQNAgent(device, logger, job_title=TITLE, lr_scheduler=LinearScheduler(0.001, 0.0, 100000),
#                  conv=True, replay_memory_size=10000, replay_warmup_length=0,
#                  C=100, minibatch_size=32, gamma=0.99, alpha=0.6,
#                  epsilon_scheduler=LinearScheduler(1.0, 0.05, 10000),
#                  beta_scheduler=LinearScheduler(0.4, 1.0, 10000),
#                  clip_grad_norm=None, update_freq=1,
#                  save_nn=False, load_nn_path=None)

# agent = DoubleDQNAgent(device, logger, job_title=TITLE, lr=0.001, conv=False,
#                  replay_memory_size=1000, replay_warmup_length=0,
#                  C=1000, minibatch_size=32, gamma=0.99,
#                  epsilon_start=1.0, epsilon_end=0.00, epsilon_decay_steps=50000,
#                  clip_grad_norm=0.5, update_freq=4,
#                  save_nn=False, load_nn_path=None)

agent = DRQNAgent(device, logger, job_title=TITLE, lr=0.0001, conv=True,
                 replay_memory_size=1000, replay_warmup_length=100,
                 C=5000, minibatch_size=32, gamma=0.99, unroll_iterations=8,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=500000,
                 clip_grad_norm=10.0, update_freq=4,
                 save_nn=True, load_nn_path=None)

# agent = DQNAgent(device, logger, job_title=TITLE, lr=0.001, conv=False,
#                  replay_memory_size=1000, replay_warmup_length=0,
#                  C=1000, minibatch_size=32, gamma=0.99,
#                  epsilon_start=1.0, epsilon_end=0.00, epsilon_decay_steps=10000,
#                  clip_grad_norm=0.5, update_freq=4,
#                  save_nn=True, load_nn_path=None)

# agent = SACAgent(device, logger, job_title=TITLE, actor_lr=0.0001, qfunc_lr=0.0001, gamma=0.99,
#                replay_memory_size=10000, minibatch_size=32, update_freq=4, alpha=0.5, target_factor=0.005,
#                decay_steps=None, decay_rate=None, save_nn=False, load_path=None,)

# agent = PPOAgent(device, logger, job_title=TITLE, actor_lr=0.0003, critic_lr=0.0003, gamma=0.99, lam=0.95,
#                conv=False, cont=True, tmax=128, epsilon=0.2, epochs=3, minibatch_size=32, 
#                decay_steps=None, decay_rate=None, entropy_weight=0.0, clip_grad_norm=None,
#                save_nn=False, load_path=None,)

# agent = A2CAgent(device, logger, job_title=TITLE, actor_lr=0.0005, critic_lr=0.0005, gamma=0.99, lam=0.96,
#                conv=False, cont=True, tmax=30, decay_steps=None, decay_rate=None,
#                entropy_weight=0.01, clip_grad_norm=0.5,
#                save_nn=False, load_path=None,)

# agent = ReinforceAgent(device, logger, job_title=TITLE, use_baseline=True, 
#                                policy_lr=0.001, state_value_lr=0.01, gamma=0.99,
#                                save_nn=False, load_path=None)

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