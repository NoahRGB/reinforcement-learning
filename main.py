
import gymnasium as gym
import minigrid

import utils
import agents
import envs

DEVICE = utils.detect_torch_device(quiet=False)
TITLE = "tests"
USE_NORMAL_LOGS = False
USE_TENSORBOARD_LOGS = True
PRINT_PROGRESS = True
NETWORK_SAVE_INTERVAL = 100
LOGGER = utils.Logger(USE_TENSORBOARD_LOGS,
                         USE_NORMAL_LOGS,
                         PRINT_PROGRESS,
                         NETWORK_SAVE_INTERVAL,
                         f"results/temps/{TITLE}",
                         [utils.Logger.Category.REWARD,
                          utils.Logger.Category.LOSS])
SEED = 1
ENV_NAME = "MiniGrid-Empty-8x8-v0" # "MiniGrid-MemoryS7-v0" # "ALE/Pong-v5"
NUM_ENVS = 1
TIMESTEPS = 2000000

# agent = agents.TD3(lr=0.001, gamma=0.98, noise_factor=0.1,
#                    replay_size=200000, minibatch_size=256,
#                    target_factor=0.005, d=2, noise_clip=0.5,
#                    warmup_steps=10000)

# agent = agents.DDPG(lr=0.001, gamma=0.98, noise_factor=0.1,
#                      replay_size=200000, minibatch_size=256, 
#                      update_freq=1, target_factor=0.005,
#                      warmup_steps=10000)

# agent = agents.SAC(lr=0.001, gamma=0.99, replay_size=1000000,
#                    minibatch_size=256, update_freq=1,
#                    alpha_start=0.0001, target_factor=0.005,
#                    warmup_steps=100)

# agent = agents.PPO(lr=0.001, gamma=0.9, lam=0.95, tmax=1024,
#                    epsilon=0.2, epochs=10, minibatch_size=256,
#                    value_weight=0.5, entropy_weight=0.0, cgn=0.5)

# agent = agents.REINFORCE(policy_lr=0.001, state_value_lr=0.01,
#                          gamma=0.99, use_baseline=True)

# agent = agents.A2C(lr=0.0007, gamma=0.9,
#                    lam=0.9, tmax=8,
#                    value_weight=0.4,
#                    entropy_weight=0.0,
#                    cgn=0.5)

# agent = agents.DRQN(lr=0.0001, replay_size=1000,
#                    C=1000, update_freq=4, minibatch_size=32,
#                    gamma=0.99, epsilon_start=1.0,
#                    epsilon_end=0.01, epsilon_steps=500000,
#                    cgn=10.0, warmup_steps=100000,
#                    unroll_iterations=4,
#                    load_path=None)

# agent = agents.PrioritisedDQN(lr=0.0001, replay_size=1000,
#                    C=1000, update_freq=4, minibatch_size=32,
#                    gamma=0.99, epsilon_start=1.0,
#                    epsilon_end=0.1, epsilon_steps=10000,
#                    cgn=10.0, warmup_steps=0,
#                    alpha=0.6, beta=0.4)

# agent = agents.DoubleDQN(lr=0.001, replay_size=100000,
#                    C=10000, update_freq=4, minibatch_size=64,
#                    gamma=0.9, epsilon_start=1.0,
#                    epsilon_end=0.05, epsilon_steps=100000,
#                    cgn=10.0, warmup_steps=64)

agent = agents.DQN(lr=0.001, replay_size=100000,
                   C=10000, update_freq=4, minibatch_size=64,
                   gamma=0.99, epsilon_start=1.0,
                   epsilon_end=0.05, epsilon_steps=300000,
                   cgn=10.0, warmup_steps=30000)


agent.to(DEVICE)
env = envs.Gymenv(ENV_NAME, NUM_ENVS, seed=SEED, normalise_obs=True, render_mode=None)
agent.learn(TIMESTEPS, env, LOGGER, seed=SEED)
 