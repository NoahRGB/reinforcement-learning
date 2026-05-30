# file for stable baselines experiments

from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import pickle
import gymnasium as gym

from environments.custom.rnn_ant.env import RNNAntEnv

gym.register(id="custom/RNNAntEnv-v0", entry_point=RNNAntEnv, max_episode_steps=30)

class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_done_count = 0

    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                self.episode_done_count += 1
                print(self.episode_done_count)
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)

        return True

log_path = "./runs"

SEED = 1
TIMESTEPS = 100000
NUM_ENVS = 10
STATS_WINDOW = 100

vec_env = make_vec_env("custom/RNNAntEnv-v0", n_envs=NUM_ENVS, seed=SEED)

agent = A2C("MlpPolicy", vec_env, ent_coef=0.001, stats_window_size=STATS_WINDOW, seed=SEED, tensorboard_log=log_path)
# agent = PPO("MlpPolicy", vec_env, stats_window_size=STATS_WINDOW, seed=SEED, tensorboard_log=log_path)

print(agent.policy)

logger = EpisodeRewardLogger()

trained_agent = agent.learn(total_timesteps=TIMESTEPS, callback=logger)

episode_rewards = np.array(logger.episode_rewards)

# with open(f"./results/experiments/sb_comparisons/cartpole/sb_seed{SEED}_envs{NUM_ENVS}.pkl", "wb") as f:
#     pickle.dump(episode_rewards, f)