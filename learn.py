import numpy as np

def learn(episodes, env, agent, resume=False, timeouts=False, quiet=True):
    agent.initialise(env.get_state_space_size(), env.get_action_space_size(), env.get_start_state(), resume=resume)
    if not quiet: print(f"Starting learning over {episodes} episodes")
    reward_history = np.empty(episodes)
    for _ in range(episodes):
        total_reward = 0
        env.reset()
        s = env.get_start_state()
        done = False
        t = 0
        while not done:
            a = agent.run_policy(s, t)
            sprime, r, done = env.step(s, a)
            agent_done = agent.update(s, sprime, a, r, done)
            if resume: print(a, sprime, r, done, agent_done)
            if agent_done and timeouts:
                total_reward = 9999
                done = True
            s = sprime
            total_reward += r
            t += 1
        agent.finish_episode()
        if not quiet: print(f"Episode {_} reward: {total_reward}")
        reward_history[_] = total_reward
    if not quiet: print(f"Finished learning over {episodes} episodes")
    return reward_history
