import pickle

with open("./results/temps/data/sb3_a2c_cartpole/mean_episodic_reward.pkl", "rb") as f:
    rewards = pickle.load(f)

print(rewards)