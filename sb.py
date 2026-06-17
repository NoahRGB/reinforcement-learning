# file for stable baselines experiments

from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from sb3_contrib import RecurrentPPO

import numpy as np
import pickle
import gymnasium as gym

from envs import POMDPCartPole

gym.register(id="POMDPCartPole", entry_point=POMDPCartPole)

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

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


LOG_PATH = "./runs"
SEED = 1
TIMESTEPS = 100000
NUM_ENVS = 8
STATS_WINDOW = 100

# env = gym.make("Pendulum-v1", render_mode=None)
vec_env = make_vec_env("POMDPCartPole", n_envs=NUM_ENVS, seed=SEED)
vec_env = VecNormalize(vec_env)

agent = RecurrentPPO("MlpLstmPolicy", vec_env, learning_rate=0.001, gamma=0.98, gae_lambda=0.8, n_steps=32,
                   ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                   n_epochs=20, batch_size=256,
                   tensorboard_log=LOG_PATH,
                   seed=SEED,
                   clip_range=linear_schedule(0.2),
                   policy_kwargs=dict(
                        ortho_init=False,
                        activation_fn=torch.nn.ReLU,
                        lstm_hidden_size=64,
                        enable_critic_lstm=True,
                        net_arch=dict(pi=[64], vf=[64])
                    ),)

# agent = TD3("MlpPolicy", vec_env, gamma=0.98, buffer_size=200000, learning_starts=10000, target_policy_noise=0.1, stats_window_size=STATS_WINDOW, seed=SEED, tensorboard_log=LOG_PATH)
# agent = A2C("MlpPolicy", vec_env, ent_coef=0.0, stats_window_size=STATS_WINDOW, seed=SEED, tensorboard_log=LOG_PATH)

print(agent.policy)

logger = EpisodeRewardLogger()

trained_agent = agent.learn(total_timesteps=TIMESTEPS, callback=logger)

# logger.save(f"./results/temps/data/sb3_a2c_cartpole_seed{SEED}_timesteps{TIMESTEPS}")