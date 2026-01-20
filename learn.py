import numpy as np

def learn(episodes, env, agent, resume=False, timeouts=False, quiet=True):
    agent.initialise(env.get_state_space_size(), env.get_action_space_size(), resume=resume)
    if not quiet: print(f"Starting learning over {episodes} episodes")
    reward_history = np.empty(episodes) 
    for _ in range(episodes):
        total_reward = 0
        env.reset()
        s = env.get_start_state()
        done = False
        while not done:
            a = agent.run_policy(s)
            sprime, r, done = env.step(s, a)
            agent_done = agent.update(s, sprime, a, r)
            if agent_done and timeouts:
                total_reward = 9999
                done = True
            s = sprime
            total_reward += r
        agent.finish_episode()
        if not quiet: print(f"Episode {_} reward: {total_reward}")
        reward_history[_] = total_reward
    if not quiet: print(f"Finished learning over {episodes} episodes")
    return reward_history
   
def evaluate(episodes, env, agent, timeout=False, quiet=True):
    if not quiet: print(f"Starting evaluation over {episodes} episodes")
    agent.toggle_eval()
    reward_history = learn(episodes, env, agent, resume=True, timeouts=timeout, quiet=True)
    agent.toggle_eval()
    if not quiet: print(f"Finished evaluation over {episodes} episodes")
    return reward_history

def parallel_learn_evaluate(episodes, env, agent, resume=False, quiet=True):
    cumulative_learning_reward_history = np.empty(episodes)
    cumulative_eval_reward_history = np.empty(episodes)
    if not resume:
        agent.initialise(env.get_state_space_size(), env.get_action_space_size())
    for _ in range(episodes):
        total_learning_reward = 0
        env.reset()
        s = env.get_start_state()
        done = False
        while not done:
            a = agent.run_policy(s)
            sprime, r, done = env.step(s, a)
            agent.update(s, sprime, a, r)
            s = sprime
            total_learning_reward += r
        agent.finish_episode()
        if not quiet: print(f"finished learning episode {_}")
        cumulative_learning_reward_history[_] = total_learning_reward
        cumulative_eval_reward_history[_] = evaluate(1, env, agent, timeout=True)[0]
        if not quiet: print(f"finished eval episode {_}")
    return cumulative_learning_reward_history, cumulative_eval_reward_history
