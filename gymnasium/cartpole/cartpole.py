import gymnasium as gym

EPISODE_COUNT = 100
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=1)
is_terminated, is_truncated = False, False

for episode_num in range(EPISODE_COUNT):

    observation, info = env.reset()

    while not is_terminated and not is_truncated:
        action = env.action_space.sample()
        observation, reward, is_terminated, is_truncated, info = env.step(action)

env.close()

