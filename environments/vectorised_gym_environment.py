from environments.environment import Environment
from environments.spaces import DiscreteSpace, ContinuousSpace, EnvType,  detect_space

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class VectorisedGymEnvironment(Environment):
    def __init__(self, name, num_envs, is_recording, **kwargs):

        self.num_envs = num_envs

        def make_one_env():
            env = gym.make(name, **kwargs)
            return env

        envs = [make_one_env for _ in range(self.num_envs)]

        self.env = gym.vector.SyncVectorEnv(envs)

        if is_recording:
            self.env = RecordVideo(self.env, ".", episode_trigger=lambda x: True)

        self.action_space = detect_space(self.env.single_action_space)
        self.state_space = detect_space(self.env.single_observation_space)

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

    def get_env_type(self):
        return EnvType.VECTORISED

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space
    
    def get_num_envs(self):
        return self.num_envs