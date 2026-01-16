import random, math, time

from utils import argmax, plot
from bandits import ActionValueBandit, GradientBandit

def run_bandit_testbed(trials, bandits, min_step=100, max_step=1000, step_step=100, quiet=True, do_plot=True):
    bandit_avg_rewards = {bandit: [] for bandit in bandits}
    bandit_avg_optimal_percentage = {bandit: [] for bandit in bandits}
    step_range = [1] + [i for i in range(min_step, max_step+1, step_step)]
    if min_step == max_step: step_range = [min_step]

    # for every k-armed bandit in the bandits list, find the average
    # reward/optimal action % obtained over the given number of trials for every
    # step in the step_range
    for i, bandit in enumerate(bandits):
        for steps in step_range:
            start_time = time.perf_counter()
            if not quiet: print(f"Running bandit {i+1} ({bandit.get_config()}) over {steps} steps...", end="", flush=True)
            avg_reward, avg_optimal_percentage = 0, 0
            for trial in range(0, trials):
                for step in range(0, steps):
                    bandit.run()
                avg_reward += sum(bandit.rewards) / steps 
                avg_optimal_percentage += (bandit.calculate_optimal_action_count() / steps) * 100 
                bandit.reset()
            bandit_avg_rewards[bandit].append(avg_reward / trials)
            bandit_avg_optimal_percentage[bandit].append(avg_optimal_percentage / trials)
            if not quiet: print(f" done ({round(time.perf_counter() - start_time, 2)}s)")
    if do_plot:
        plot(step_range, list(bandit_avg_rewards.values()), y_labels=[bandit.get_config() for bandit in bandits], x_title="Number of steps run", y_title=f"Average reward recieved over {trials} trials")
        plot(step_range, list(bandit_avg_optimal_percentage.values()), y_lims=[0, 100], x_title="Number of steps run", y_title=f"Average percentage of steps that chose the optimal action over {trials} trials", y_labels=[bandit.get_config() for bandit in bandits])
    return bandit_avg_rewards, bandit_avg_optimal_percentage

def run_parameter_study():
    all_x_vals = [math.pow(2, i) for i in range(-7, 3, 1)]
    x_vals = [[math.pow(2, i) for i in range(-7, -1, 1)], [math.pow(2, i) for i in range(-4, 3, 1)], [math.pow(2, i) for i in range(-2, 3, 1)], [math.pow(2, i) for i in range(-5, 0, 1)]]
    x_ticks = [list(range(-7, -1, 1)), list(range(-4, 3, 1)), list(range(-2, 3, 1)), list(range(-5, 0, 1))]
    bandits = [[ActionValueBandit(10, 0, epsilon=param) for param in x_vals[0]],
            [ActionValueBandit(10, 0, c=param, step_size=0.1) for param in x_vals[1]],
            [ActionValueBandit(10, param, epsilon=0, step_size=0.1) for param in x_vals[2]],
            [GradientBandit(10, param, use_baseline=True) for param in x_vals[3]]]
               
    y_vals = []
    for bandit_set in bandits:
        avg_rewards, avg_optimals = run_bandit_testbed(1000, bandit_set, min_step=1000, max_step=1000, step_step=1, quiet=False, do_plot=False)
        y_vals.append([val[0] for val in avg_rewards.values()])
    
    plot(x_ticks, y_vals, save_path="parameter_study", x_title="Parameter value for each algorithm (Q1,ε,c,a)", y_title="Avg reward over 2000 trials of 1000 steps", y_labels=["εgreedy", "UCB", "optimistic greedy", "gradient"], xticks=range(-7, 3), xticks_labels=[f"{num}/{den}" for (num, den) in [x.as_integer_ratio() for x in all_x_vals]])
                 
if __name__ == "__main__":
    # action value bandits at different epsilon values
    #bandits = [ActionValueBandit(10, 0, epsilon=epsilon) for epsilon in [0.0, 0.01, 0.1]]

    # optimistic+greedy vs realistic+ε-greedy
    #bandits = [ActionValueBandit(10, 5, epsilon=0.0, step_size=0.1), ActionValueBandit(10, 0, epsilon=0.1, step_size=0.1)]

    # epsilon-greedy vs UCB 
    #bandits = [ActionValueBandit(10, 0, epsilon=0.1, step_size=0.1), ActionValueBandit(10, 0, c=2, step_size=0.1)] 
    
    # gradient bandits with/without baselines at different step sizes
    # bandits = [GradientBandit(10, 0.1, q_centre=4), GradientBandit(10, 0.1, q_centre=4, use_baseline=True), GradientBandit(10, 0.4, q_centre=4), GradientBandit(10, 0.4, q_centre=4, use_baseline=True)]
    
    #run_bandit_testbed(2000, bandits, quiet=False)
    run_parameter_study()
