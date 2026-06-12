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

# paths = ["dqn_2frames", "dqn_4frames", "dqn_1frames"]
# labels = ["DQN 2 frames", "DQN 4 frames", "DQN 1 frame"]

paths = ["me_td3_pendulum", "sb3_td3_pendulum"]
labels = ["My TD3", "SB3 TD3"]

for path, label in zip(paths, labels):
    all_seed_timesteps = []
    all_seed_rewards = []
    for seed in seeds:
        seed_timesteps, seed_rewards = load_seed(f"./results/temps/data/{path}_seed{seed}/episodic_reward.pkl")
        all_seed_timesteps.append(seed_timesteps)
        all_seed_rewards.append(seed_rewards)

    min_timesteps = min(timesteps[-1] for timesteps in all_seed_timesteps)
    grid = np.linspace(0, min_timesteps, 500)

    curves = []
    for t, r in zip(all_seed_timesteps, all_seed_rewards):
        curves.append(np.interp(grid, t, r))

    curves = np.array(curves)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    plt.plot(grid, mean, label=label)
    plt.fill_between(grid, mean - std, mean + std, alpha=0.2)

plt.xlabel("Timesteps")
plt.ylabel("Episodic Reward")
plt.legend()
plt.title("Pong with different frame stack sizes, averaged over 10 trials")
# plt.xticks([0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6], ["0M", "0.5M", "1M", "1.5M", "2M", "2.5M", "3M"])
# plt.savefig("results/temps/data/pong_frame_stack_comparison.png", dpi=300, bbox_inches="tight")
plt.show()