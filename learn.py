import numpy as np

def learn(episodes, env, agent, resume=False, timeouts=False, quiet=True):
    state_space = env.get_state_space()
    action_space = env.get_action_space()
    if type(state_space) not in agent.get_supported_state_spaces():
        print("Envrionment state space not supported by agent")
        return False
    if type(action_space) not in agent.get_supported_action_spaces():
        print("Envrionment action space not supported by agent")
        return False

    agent.initialise(state_space, action_space, env.get_start_state(), resume=resume)
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
