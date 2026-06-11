import numpy as np
import pickle
import matplotlib.pyplot as plt

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

all_data = []

def load_seed(path):
    with open(f"{path}", "rb") as f:
        data = pickle.load(f)
    rewards, timesteps = zip(*data)
    return np.array(timesteps), np.array(rewards)

min_timesteps = 296952
grid = np.linspace(0, min_timesteps, 500)

sb_curves = []
me_curves = []
for seed in seeds:
    me_t, me_r = load_seed(f"./results/temps/data/me_a2c_cartpole_seed{seed}/episodic_reward.pkl")
    me_curves.append(np.interp(grid, me_t, me_r))
    sb_t, sb_r = load_seed(f"./results/temps/data/sb3_a2c_cartpole_seed{seed}/episodic_reward.pkl")
    sb_curves.append(np.interp(grid, sb_t, sb_r))

sb_curves = np.array(sb_curves)
sb_mean = sb_curves.mean(axis=0)
sb_std = sb_curves.std(axis=0)

me_curves = np.array(me_curves)
me_mean = me_curves.mean(axis=0)
me_std = me_curves.std(axis=0)


plt.plot(grid, sb_mean, label="SB3 A2C")
plt.fill_between(grid, sb_mean - sb_std, sb_mean + sb_std, alpha=0.2)
plt.plot(grid, me_mean, label="My A2C")
plt.fill_between(grid, me_mean - me_std, me_mean + me_std, alpha=0.2)
plt.xlabel("Timesteps")
plt.ylabel("Episodic Reward")
plt.legend()
plt.show()