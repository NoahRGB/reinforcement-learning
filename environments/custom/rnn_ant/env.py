import gymnasium as gym
from gymnasium import spaces

import numpy as np

class RNNAntEnv(gym.Env):

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.randomise_food = True
        self.num_memory_nodes = 0
        self.state_dimension = 3 + self.num_memory_nodes
        self.speed = 0.2

        self.observation_space = spaces.Box(np.full((3 + self.num_memory_nodes), -10),
                                            np.full((3 + self.num_memory_nodes), 10),
                                            shape=(3 + self.num_memory_nodes,), dtype=np.float32)

        self.action_space = spaces.Box(np.full((2 + self.num_memory_nodes), -10),
                                       np.full((2 + self.num_memory_nodes), 10),
                                       shape=(2 + self.num_memory_nodes,), dtype=np.float32)
        
        self.initial_food_location = (np.random.rand(2) - 0.5) * (8 if self.randomise_food else 0)
        self.initial_state = np.concatenate([
            (np.random.rand(2) - 0.5) * 10,
            np.zeros((self.state_dimension - 2))
        ], axis=0)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.food_location = self.initial_food_location.copy()
        self.state = self.initial_state.copy()

        return self.state, {}
    
    def food_density(self, pos):
        # returns the amount of food at the given pos
        bump_width = 8
        return np.exp(np.sum(-(pos - self.food_location)**2, axis=0, keepdims=True) / bump_width).squeeze()

    def sensor_calculation(self, pos):
        sensor_result = self.food_density(pos)
        return sensor_result
    
    def run_one_step_of_physics_model(self, action):
        pos = self.state[0:2]
        a = action[0:2]
        new_pos = pos + a * self.speed
        sprime = new_pos.tolist()

        sens = self.sensor_calculation(new_pos)

        sprime.append(sens.tolist())

        for i in range(0, self.num_memory_nodes):
            sprime.append(action[2+i:3+i].tolist()[0])

        rewards = self.food_density(sprime[:2])
        return [rewards, sprime]

    def step(self, action):

        [rewards, s] = self.run_one_step_of_physics_model(action)
        self.state = s

        return s, rewards, False, False, {}