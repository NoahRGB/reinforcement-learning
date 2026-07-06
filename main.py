import sys
import gymnasium as gym

import utils
import agents
import envs

gym.register(id="POMDPCartPole", entry_point=envs.POMDPCartPole)

DEVICE = utils.detect_torch_device(quiet=False)
USE_NORMAL_LOGS = False
USE_TENSORBOARD_LOGS = True
PRINT_PROGRESS = True
NETWORK_SAVE_INTERVAL = 0
SEED = 1
ENV_NAME = "Pendulum-v1" # "CartPole-v1" # "MiniGrid-MemoryS7-v0" # 
NUM_ENVS = 4
TIMESTEPS = 1000000
TITLE = f"a"

LOGGER = utils.Logger(USE_TENSORBOARD_LOGS,
                         USE_NORMAL_LOGS,
                         PRINT_PROGRESS,
                         NETWORK_SAVE_INTERVAL,
                         f"results/temps/{TITLE}",
                         [utils.Logger.Category.REWARD,
                          utils.Logger.Category.LOSS])


# agent = agents.TD3(lr=0.001, gamma=0.98, noise_factor=0.1, target_noise_factor=0.2,
#                    replay_size=200000, minibatch_size=256,
#                    target_factor=0.005, d=2, noise_clip=0.5,
#                    warmup_steps=10000, gradient_steps=1)

# agent = agents.DDPG(lr=0.001, gamma=0.98, noise_factor=0.1,
#                      replay_size=200000, minibatch_size=256, 
#                      update_freq=1, target_factor=0.005,
#                      warmup_steps=10000, gradient_steps=1)

# agent = agents.SAC(lr=0.001, gamma=0.99, replay_size=200000,
#                    minibatch_size=256, update_freq=1,
#                    alpha_start=0.0001, target_factor=0.005,
#                    warmup_steps=10000, gradient_steps=1)

# agent = agents.LSTM_PPO(lr_scheduler=utils.LinearScheduler(0.001, 0.0, 100000), gamma=0.98, lam=0.8, tmax=32,
#                    epsilon_scheduler=utils.LinearScheduler(0.2, 0.0, 100000),
#                    epochs=20, minibatch_size=8,
#                    value_weight=0.5, entropy_weight=0.0, 
#                    cgn=0.5, lstm_hidden_size=64)

agent = agents.PPO(lr_scheduler=utils.LinearScheduler(0.0003, 0.0003, 1), 
                   gamma=0.9, lam=0.95, tmax=1024, epsilon=0.2, epochs=10, 
                   minibatch_size=32, value_weight=0.5, entropy_weight=0.0, 
                   cgn=0.5, load_path=None)

# agent = agents.REINFORCE(policy_lr=0.01, state_value_lr=0.01,
#                          gamma=0.99, use_baseline=True)

# agent = agents.A2C(lr=0.001, gamma=0.99,
#                    lam=0.95, tmax=5,
#                    value_weight=0.2,
#                    entropy_weight=0.01,
#                    cgn=10.0)

# agent = agents.NewDRQN(lr_scheduler=utils.LinearScheduler(0.001, 0.0, 50000), replay_size=10000,
#                    C=300, update_freq=4, minibatch_size=32, gamma=0.9,
#                    epsilon_scheduler=utils.LinearScheduler(0.01, 0.01, 1),
#                    cgn=10.0, warmup_steps=1000,
#                    unroll_iterations=20, gradient_steps=1, lstm_size=64,
#                    load_path=None)

# agent = agents.DRQN(lr_scheduler=utils.LinearScheduler(0.001, 0.0, 50000), replay_size=10000,
#                    C=300, update_freq=4, minibatch_size=32, gamma=0.9,
#                    epsilon_scheduler=utils.LinearScheduler(0.01, 0.01, 1),
#                    cgn=10.0, warmup_steps=1000,
#                    unroll_iterations=20, gradient_steps=1, lstm_size=64,
#                    load_path=None)

# agent = agents.RainbowDQN(lr=0.001, replay_size=10000,
#                    C=500, update_freq=4,
#                    minibatch_size=64, gamma=0.95, 
#                    cgn=10.0, warmup_steps=1000, gradient_steps=1,
#                    vmin=0, vmax=100, N=10, nstep=5, alpha=0.5, 
#                    beta_scheduler=utils.LinearScheduler(0.4, 1.0, 10000),
#                    epsilon_scheduler=utils.LinearScheduler(1.0, 0.05, 1),
#                    use_distributional=False, use_noisy=True, use_dueling=True,
#                    use_double=True, use_per=True, load_path=None)

# agent = agents.R2D2(lr=0.001, replay_size=10000,
#                    C=100, update_freq=4, minibatch_size=32, 
#                    gamma=0.99, epsilon_scheduler=utils.LinearScheduler(1.0, 0.05, 5000),
#                    cgn=10.0, warmup_steps=1000, gradient_steps=1, seq_len=20, overlap=20, eta=0.9,
#                    alpha=0.5, beta_scheduler=utils.LinearScheduler(0.6, 1.0, 10000), nsteps=5,
#                    lstm_size=128, use_dueling=True, use_double=True, use_per=False, load_path=None)

# agent = agents.CuriousDQN(lr=0.001, replay_size=10000,
#                    C=500, update_freq=4, 
#                    minibatch_size=64, gamma=0.99, 
#                    epsilon_scheduler=utils.LinearScheduler(1.0, 0.05, 10000),
#                    cgn=10.0, warmup_steps=1000, gradient_steps=1,
#                    curiosity_weight=0.01, load_path=None)

# agent = agents.DQN(lr=0.001, replay_size=10000,
#                    C=100, update_freq=4, 
#                    minibatch_size=64, gamma=0.9, 
#                    epsilon_scheduler=utils.LinearScheduler(1.0, 0.05, 10000),
#                    cgn=10.0, warmup_steps=1000, gradient_steps=1,
#                    load_path=None)

agent.to(DEVICE)
env = envs.Gymenv(ENV_NAME, NUM_ENVS, seed=SEED, normalise_obs=False, render_mode=None) # , allowed_actions=[0, 1, 2], swap_channel=True
agent.learn(TIMESTEPS, env, LOGGER, seed=SEED)
 