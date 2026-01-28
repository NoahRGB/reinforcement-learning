from environments.environment import Environment
from environments.spaces import DiscreteSpace, ContinuousSpace

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class GymEnvironment(Environment):
    def __init__(self, name, is_recording, **kwargs):
        if is_recording and kwargs["render_mode"] == None:
            kwargs["render_mode"] = "rgb_array"

        self.env = gym.make(name, **kwargs)
        if is_recording:
            self.env = RecordVideo(self.env, ".", episode_trigger=lambda x: True)

        if type(self.env.observation_space) == gym.spaces.Discrete:
            self.state_space = DiscreteSpace(self.env.observation_space.n)
        elif type(self.env.observation_space) == gym.spaces.Box:
            gym_box = self.env.observation_space
            self.state_space = ContinuousSpace(gym_box.shape[0], gym_box.low, gym_box.high)

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
