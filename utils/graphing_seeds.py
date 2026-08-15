import numpy as np
import pickle
import matplotlib.pyplot as plt

# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
seeds = [1, 2, 3, 4, 5, 6, 7]

all_data = []

def load_seed(path):
    with open(f"{path}", "rb") as f:
        data = pickle.load(f)
    rewards, timesteps = zip(*data)
    return np.array(timesteps), np.array(rewards)


# paths = ["ppo_hopper"]
# labels = ["PPO"]

paths = ["ppo_humanoid"]
labels = ["PPO"]


for path, label in zip(paths, labels):
    all_seed_timesteps = []
    all_seed_rewards = []
    for seed in seeds:
        try:
            seed_timesteps, seed_rewards = load_seed(f"../results/robot_comparisons/humanoid/{path}_seed{seed}/episodic_reward.pkl")
            # seed_timesteps, seed_rewards = load_seed(f"../results/atari_comparisons/ppo_dqn_boxing/{path}_seed{seed}/episodic_reward.pkl")
            # seed_timesteps, seed_rewards = load_seed(f"../results/rl_sb_comparisons/sac_pendulum/{path}_seed{seed}/episodic_reward.pkl")
            # seed_timesteps, seed_rewards = load_seed(f"./results/temps/data/{path}_seed{seed}/episodic_reward.pkl")

            all_seed_timesteps.append(seed_timesteps)
            all_seed_rewards.append(seed_rewards)
        except FileNotFoundError:
            print(f"File not found for {path}_seed{seed}")

    min_timesteps = min(timesteps[-1] for timesteps in all_seed_timesteps)
    # min_timesteps=4000000
    grid = np.linspace(0, min_timesteps, 450)

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
# plt.ylim(0, 850)
# plt.xlim(0, 100000)
plt.title("DQN vs PPO on Boxing, averaged over 10 trials")
# plt.xticks([0, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 10e6], ["0", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M"])
# plt.savefig("results/temps/data/boxing_comparison.png", dpi=300, bbox_inches="tight")
plt.show()