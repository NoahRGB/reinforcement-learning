import gymnasium as gym

class POMDPCartPole(gym.Env):
    def __init__(self, render_mode=None):

        self.env = gym.make("CartPole-v1", render_mode=render_mode)

        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[[0, 2]],
            high=self.env.observation_space.high[[0, 2]],
            dtype=self.env.observation_space.dtype
        )

        self.action_space = self.env.action_space
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation[[0, 2]], info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[[0, 2]], reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()