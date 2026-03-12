import numpy as np

from agents.agent import Agent
from environments.environment import Environment
from utils.evaluate import evaluate

def learn(episodes, env: Environment, agent: Agent, eval_period=1, resume=False, quiet=True):
    # check agent-environment compatibility
    state_space = env.get_state_space()
    action_space = env.get_action_space()
    if type(state_space) not in agent.get_supported_state_spaces():
        print("Envrionment state space not supported by agent")
        return False
    if type(action_space) not in agent.get_supported_action_spaces():
        print("Envrionment action space not supported by agent")
        return False

    # initialise
    agent.initialise(state_space, action_space, env.get_start_state(), resume=resume)
    if not quiet: print(f"Starting learning over {episodes} episodes")
    reward_history = np.empty(episodes)

    # training loop
    for episode_num in range(episodes):
        # reset episode
        total_reward = 0
        env.reset()
        s = env.get_start_state()
        done = False
        t = 0
        
        # until episode is terminated
        while not done:
            # gather and execute a, receive s', r
            a = agent.run_policy(s, t)
            sprime, r, done = env.step(s, a)
            agent.update(s, sprime, a, r, done)
            s = sprime
            total_reward += r
            t += 1
        agent.finish_episode(episode_num)

        # after X episodes, run an eval episode
        if eval_period > 0 and episode_num % eval_period == 0:
            evaluate(agent, env, True)

        # log progress + reward
        if not quiet: print(f"Episode {episode_num} reward: {total_reward}")
        reward_history[episode_num] = total_reward
    if not quiet: print(f"Finished learning over {episodes} episodes")

    return reward_history
