import gymnasium as gym

from env import BallGame

gym.register(id="gymnasium_env/BallGame-v0", entry_point=BallGame, max_episode_steps=300)

env = gym.make("gymnasium_env/BallGame-v0", render_mode="human")

observation, info = env.reset()

for _ in range(1000):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        

env.close()