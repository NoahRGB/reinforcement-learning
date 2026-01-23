import random

import matplotlib.pyplot as plt

def argmax(d):
    # returns a random key in the dict among those that have the max value
    max_val = max(list(d.values())) 
    maxs = [key for (key, val) in d.items() if val == max_val]
    return random.choice(maxs) 

def plot(x, y_vals, y_labels=[], x_title="", y_title="", y_lims=None, x_lims=None, xticks=None, xticks_labels=None, save_path=None):
    if not isinstance(x[0], list):
        for i in range(0, len(y_vals)):
            plt.plot(x[:len(y_vals[i])], y_vals[i], label=y_labels[i] if len(y_labels) > i else "")
    else:
        for i in range(0, len(x)):
            plt.plot(x[i], y_vals[i], label=y_labels[i] if len(y_labels) > i else "")
    plt.legend()
    if y_lims: plt.ylim(y_lims[0], y_lims[1])
    if x_lims: plt.xlim(x_lims[0], x_lims[1])
    if xticks and xticks_labels: plt.xticks(xticks, xticks_labels)
    if save_path: plt.savefig(save_path)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


