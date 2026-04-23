import pickle, os, sys

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from utils import smoothing 

smooting = float(sys.argv[1]) if len(sys.argv) > 1 else 0.99

loc = "./results/temps/data/"
# loc = "./results/experiments/pong_decay/"
cols = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "black", "brown", "pink", "gray", "olive", "cyan", "navy", "teal", "maroon", "lime", "coral", "gold"]
data = []
files = [f for f in os.listdir(loc) if f.endswith(".pkl")]
labels = [file[:-4].split("_")[-1] for file in files]

for file_idx, file in enumerate(files):
    with open(loc + file, "rb") as f:
        try:
            new_data = np.array(pickle.load(f))
            data.append(new_data)
            plt.plot(smoothing(new_data, smooting), alpha=0.9, color=cols[file_idx], label=labels[file_idx])
        except Exception as e:
            print(f"Error loading {file}: {e}")

plt.xlabel("Episode")
plt.ylabel("Episode reward")
# plt.title("How the number of episodes over which epsilon is decayed affects performance of DQN on Pong")
plt.legend()
# plt.savefig("./results/bundles/hpc/pong_decay/pong_decay.png", dpi=300, bbox_inches="tight")
plt.show()