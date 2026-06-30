
import gymnasium as gym
import minigrid

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
ENV_NAME = "POMDPCartPole" # "PongNoFrameskip-v4"
NUM_ENVS = 1
TIMESTEPS = 200000
TITLE = f"tests"

LOGGER = utils.Logger(USE_TENSORBOARD_LOGS,
                         USE_NORMAL_LOGS,
                         PRINT_PROGRESS,
                         NETWORK_SAVE_INTERVAL,
                         f"results/temps/{TITLE}",
                         [utils.Logger.Category.REWARD,
                          utils.Logger.Category.LOSS])


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

# agent = agents.LSTM_PPO(lr_scheduler=utils.LinearScheduler(0.001, 0.0, 100000), gamma=0.98, lam=0.8, tmax=32,
#                    epsilon_scheduler=utils.LinearScheduler(0.2, 0.0, 100000),
#                    epochs=20, minibatch_size=8,
#                    value_weight=0.5, entropy_weight=0.0, 
#                    cgn=0.5, lstm_hidden_size=64)

# agent = agents.PPO(lr=0.001, gamma=0.98, lam=0.8, tmax=32,
#                    epsilon=0.2, epochs=20, minibatch_size=256,
#                    value_weight=0.5, entropy_weight=0.0, cgn=0.5)

# agent = agents.REINFORCE(policy_lr=0.001, state_value_lr=0.01,
#                          gamma=0.99, use_baseline=True)

# agent = agents.A2C(lr=0.0007, gamma=0.99,
#                    lam=1.0, tmax=5,
#                    value_weight=0.5,
#                    entropy_weight=0.0,
#                    cgn=0.5)

# agent = agents.DRQN(lr=0.001, replay_size=10000,
#                    C=1000, update_freq=4, minibatch_size=32, gamma=0.99,
#                    epsilon_scheduler=utils.LinearScheduler(1.0, 0.01, 15000),
#                    cgn=10.0, warmup_steps=1000,
#                    unroll_iterations=5, gradient_steps=1, lstm_size=64,
#                    load_path=None)

# agent = agents.RainbowDQN(lr=0.0023, replay_size=100000,
#                    C=10, update_freq=256,
#                    minibatch_size=64, gamma=0.99, 
#                    cgn=10.0, warmup_steps=1000, gradient_steps=128,
#                    vmin=0, vmax=100, N=10, nstep=1, alpha=0.5, 
#                    beta_scheduler=utils.LinearScheduler(0.4, 1.0, 16000),
#                    epsilon_scheduler=utils.LinearScheduler(1.0, 0.04, 16000),
#                    use_distributional=True, use_noisy=True, use_dueling=True,
#                    use_double=True, use_per=False, load_path=None)

agent = agents.R2D2(lr=0.001, replay_size=10000,
                   C=1000, update_freq=4, minibatch_size=32, 
                   gamma=0.99, epsilon_scheduler=utils.LinearScheduler(1.0, 0.01, 15000),
                   cgn=10.0, warmup_steps=1000, gradient_steps=1, seq_len=4, overlap=2, eta=0.9,
                   alpha=0.5, beta_scheduler=utils.LinearScheduler(0.6, 0.6, 1), nsteps=2,
                   lstm_size=64, use_dueling=True, use_double=True, use_per=True, load_path=None)

# agent = agents.DQN(lr=0.001, replay_size=10000,
#                    C=1000, update_freq=4, 
#                    minibatch_size=32, gamma=0.99, 
#                    epsilon_scheduler=utils.LinearScheduler(1.0, 0.01, 15000),
#                    cgn=10.0, warmup_steps=0, gradient_steps=1,
#                    load_path=None)

agent.to(DEVICE)
env = envs.Gymenv(ENV_NAME, NUM_ENVS, seed=SEED, normalise_obs=True, render_mode=None)
agent.learn(TIMESTEPS, env, LOGGER, seed=SEED)
 