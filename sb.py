# file for stable baselines experiments

from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import pickle
import gymnasium as gym

class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.reward_history = []
        self.episode_done_count = 0
        self.vars = {
            "episodic_reward": [],
            "mean_episodic_reward": []
        }

    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                self.episode_done_count += 1
                reward = info["episode"]["r"]
                timestep = self.num_timesteps

                self.reward_history.append(reward)
                self.vars["episodic_reward"].append((reward, timestep))
                self.vars["mean_episodic_reward"].append((np.mean(self.reward_history[-100:]), timestep))

                print(f"episode {self.episode_done_count}, timesteps {timestep}, reward: {reward}")
        return True

    def save(self, path):
        import pathlib, pickle
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        for var_name, value in self.vars.items():
            with open(f"{path}/{var_name}.pkl", "wb") as f:
                pickle.dump(value, f)


LOG_PATH = "./runs"
SEED = 7
TIMESTEPS = 300000
NUM_ENVS = 8
STATS_WINDOW = 100

vec_env = make_vec_env("CartPole-v1", n_envs=NUM_ENVS, seed=SEED)


agent = A2C("MlpPolicy", vec_env, ent_coef=0.0, stats_window_size=STATS_WINDOW, seed=SEED, tensorboard_log=LOG_PATH)

logger = EpisodeRewardLogger()

trained_agent = agent.learn(total_timesteps=TIMESTEPS, callback=logger)

logger.save(f"./results/temps/data/sb3_a2c_cartpole_seed{SEED}_timesteps{TIMESTEPS}")