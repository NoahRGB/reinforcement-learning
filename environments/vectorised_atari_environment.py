from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation, ClipReward, RecordVideo
from environments.environment import Environment
from environments.spaces import DiscreteSpace, ContinuousSpace

import gymnasium as gym
import ale_py

class VectorisedAtariEnvironment(Environment):
    def __init__(self, num_envs, name, **kwargs):

        self.num_envs = num_envs

        def make_one_env():
            env = gym.make(name, frameskip=1, **kwargs)
            env = AtariPreprocessing(env,
                noop_max=0, frame_skip=4, terminal_on_life_loss=False,
                screen_size=84, grayscale_obs=True, grayscale_newaxis=False
            )
            env = FrameStackObservation(env, stack_size=4)
            return env

        envs = [make_one_env for _ in range(self.num_envs)]

        self.env = gym.vector.SyncVectorEnv(envs)
        self.action_space = DiscreteSpace(self.env.single_action_space.n)
        self.state_space = ContinuousSpace(self.env.observation_space.shape[0], self.env.observation_space.low, self.env.observation_space.high)

        # self.env = RecordVideo(self.env, ".", episode_trigger=lambda x: True)

    def __del__(self):
        self.env.close()

    def step(self, s, a):
        sprime, r, is_terminated, is_truncated, info = self.env.step(a)
        return sprime, r, (is_terminated|is_truncated)

    def reset(self):
        self.env.reset()

    def get_start_state(self):
        start_state, _ = self.env.reset()
        return start_state

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

