from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
from environments.environment import Environment
from environments.spaces import DiscreteSpace, ContinuousSpace

import gymnasium as gym
import ale_py

class AtariEnvironment(Environment):
    def __init__(self, name, **kwargs):

        self.env = gym.make(name, **kwargs)
        self.env = AtariPreprocessing(self.env,
            noop_max=10, frame_skip=1, terminal_on_life_loss=True,
            screen_size=84, grayscale_obs=True, grayscale_newaxis=False
        )
        self.env = FrameStackObservation(self.env, stack_size=4)

        if type(self.env.observation_space) == gym.spaces.Discrete:
            self.state_space = DiscreteSpace(self.env.observation_space.n)
        elif type(self.env.observation_space) == gym.spaces.Box:
            gym_box = self.env.observation_space
            if len(gym_box.shape) == 1:
                self.state_space = ContinuousSpace(gym_box.shape[0], gym_box.low, gym_box.high)
            else:
                self.state_space = ContinuousSpace(gym_box.shape, gym_box.low, gym_box.high)

        if type(self.env.action_space) == gym.spaces.Discrete:
            self.action_space = DiscreteSpace(self.env.action_space.n)
        elif type(self.env.action_space) == gym.spaces.Box:
            gym_box = self.env.action_space
            self.action_space = ContinuousSpace(gym_box.shape[0], gym_box.low, gym_box.high)

    def __del__(self):
        self.env.close()

    def step(self, s, a):
        sprime, r, is_terminated, is_truncated, info = self.env.step(a)
        return sprime, r, (is_terminated or is_truncated)

    def reset(self):
        self.env.reset()

    def get_start_state(self):
        start_state, _ = self.env.reset()
        return start_state

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

