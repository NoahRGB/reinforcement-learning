from environments.environment import Environment
from environments.spaces import DiscreteSpace, ContinuousSpace, detect_space

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class GymEnvironment(Environment):
    def __init__(self, name, is_recording, **kwargs):
        if is_recording and kwargs["render_mode"] == None:
            kwargs["render_mode"] = "rgb_array"

        self.env = gym.make(name, **kwargs)
        if is_recording:
            self.env = RecordVideo(self.env, ".", episode_trigger=lambda x: True)

        self.state_space = detect_space(self.env.observation_space)
        self.action_space = detect_space(self.env.action_space)

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
