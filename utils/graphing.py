import pickle, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def smoothing(vals, factor):
    # https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
    last_smoothed_val = vals[0]
    smoothed_vals = np.zeros(len(vals))
    for i, val in enumerate(vals):
        smoothed_val = last_smoothed_val * factor + (1 - factor) * val
        smoothed_vals[i] = smoothed_val
        last_smoothed_val = smoothed_val
    return smoothed_vals

smoothing_val = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
loc = str(sys.argv[2]) if len(sys.argv) > 2 else "./results/temps/data/"
found_pkls = []
found_dirs = [loc]

# find all .pkls in all subdirectories of loc
while len(found_dirs) > 0:
    current_dir = found_dirs.pop(0)
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path):
            found_dirs.append(item_path)
        elif item.endswith(".pkl"):
            found_pkls.append(item_path)

cols = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "black", "brown", "pink", "gray", "olive", "cyan", "navy", "teal", "maroon", "lime", "coral", "gold"]
# labels = [file[:-4].split("_")[-1] for file in found_pkls]
labels = [file[:-4].split("/")[-1] for file in found_pkls]

# if the file can be read, plot it
for file_idx, file in enumerate(found_pkls):
    with open(file, "rb") as f:
        try:
            raw_dat = pickle.load(f)
            if isinstance(raw_dat[0], tuple):
                xs = [x[1] for x in raw_dat]
                ys = [x[0] for x in raw_dat]
                plt.plot(xs, smoothing(ys, smoothing_val), alpha=0.9, color=cols[file_idx], label=labels[file_idx])
            else:
                new_data = np.array(raw_dat)
                plt.plot(smoothing(new_data, smoothing_val), alpha=0.9, color=cols[file_idx], label=labels[file_idx])
        except Exception as e:
            print(f"Error loading {file}: {e}")

plt.xlabel("Episode")
plt.ylabel("Reward")
# plt.title("")
plt.legend()
# plt.savefig("", dpi=300, bbox_inches="tight")
plt.show()