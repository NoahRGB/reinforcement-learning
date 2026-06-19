import pickle
import numpy as np
import matplotlib.pyplot as plt

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]

def load(path):
    with open(f"{path}", "rb") as f:
        return pickle.load(f)
    
# paths = ["../results/ants/rand/a2c_lstm_randfood.pkl", "../results/ants/rand/BPTT_randfood.pkl", "../results/ants/rand/ppo_lstm_randfood.pkl"]
# labels = ["A2C + LSTM", "BPTT", "PPO + LSTM"]

paths = ["../results/ants/rand/ppo_envmem_randfood.pkl", "../results/ants/rand/ppo_lstm_randfood.pkl", "../results/ants/rand/BPTT_randfood.pkl"]

labels = ["PPO + env memory", "PPO + LSTM", "BPTT"]

for i, path in enumerate(paths):
    data = load(path)
    plt.plot(data, label=labels[i])

plt.title("Random food, fixed starting position, averaged over 20 trials")
plt.xlabel("Episodes")
plt.ylabel("Avg reward")
plt.legend()
plt.savefig("./results/vids/ppo_comparison.png", dpi=300, bbox_inches="tight")
plt.show()