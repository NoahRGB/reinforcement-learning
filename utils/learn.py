import numpy as np

from agents.agent import Agent
from environments.environment import Environment
from environments.spaces import EnvType

def learn(episodes: int, env: Environment, agent: Agent, quiet=True):
    # check agent-environment compatibility
    state_space = env.get_state_space()
    action_space = env.get_action_space()
    env_type = env.get_env_type()
    if type(state_space) not in agent.get_supported_state_spaces():
        print(f"Environment state space {state_space} not supported by agent")
        return False
    if type(action_space) not in agent.get_supported_action_spaces():
        print(f"Environment action space {action_space} not supported by agent")
        return False
    if env_type not in agent.get_supported_env_types():
        print(f"Environment type {env_type} not supported by agent")
        return False

    if env_type == EnvType.VECTORISED:
        return learn_vectorised(episodes, env, agent, quiet)
    else:
        return learn_singular(episodes, env, agent, quiet)


def learn_singular(episodes: int, env: Environment, agent: Agent, quiet=True):
    # initialise
    agent.initialise(env.get_state_space(), env.get_action_space(), env.get_start_state(), env.get_num_envs())
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

        # log progress + reward
        if not quiet: print(f"Episode {episode_num} reward: {total_reward}")
        reward_history[episode_num] = total_reward
    if not quiet: print(f"Finished learning over {episodes} episodes")

    return reward_history

def learn_vectorised(episodes: int, env: Environment, agent: Agent, quiet=True):
    
    episodes_completed = 0
    agent.initialise(env.get_state_space(), env.get_action_space(), env.get_start_state(), env.get_num_envs())
    current_states = env.get_start_state() # (num_envs, state_dim)

    while episodes_completed < episodes:

        actions = agent.run_policy(current_states, -1) # (num_envs,)
        sprimes, rewards, dones = env.step(current_states, actions) # (num_envs, state_dim), (num_envs,), (num_envs,)
        agent.update(current_states, sprimes, actions, rewards, dones)
        current_states = sprimes

        num_completed_episodes = dones.sum()
        episodes_completed += num_completed_episodes
        if num_completed_episodes > 0 and not quiet:
            print(f"Completed {episodes_completed}/{episodes} episodes")