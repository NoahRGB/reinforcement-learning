import gymnasium as gym

from env import RNNAntEnv

gym.register(id="custom/RNNAntEnv-v0", entry_point=RNNAntEnv, max_episode_steps=300)

env = gym.make("custom/RNNAntEnv-v0", render_mode=None)

observation, info = env.reset()

for _ in range(1):

    action = env.action_space.sample()

    new_observation, reward, terminated, truncated, info = env.step(action)
    print("\n\n\n")
    print(f"state: {observation}\n action: {action}\n reward: {reward}\n sprime: {new_observation}")

    observation = new_observation

    if terminated or truncated:
        observation, info = env.reset()
        

env.close()