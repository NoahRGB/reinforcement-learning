import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from utils import smoothing 

data = pd.read_csv("./graph_data/pong_avg_reward.csv").iloc[:, -1].to_numpy()

plt.plot(data, color="red")
plt.xlabel("Episode")
plt.ylabel("Mean reward over the last 100 episodes")
plt.savefig("pong_avg_reward.png", dpi=500, bbox_inches="tight")
plt.show()
