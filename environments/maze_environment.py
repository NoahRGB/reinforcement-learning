from environments.environment import Environment
from environments.spaces import DiscreteSpace

import numpy as np

class MazeEnvironment(Environment):
    def __init__(self):
        # width = 22, height = 14
        self.maze = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ])
        self.maze_width = self.maze.shape[1]
        self.maze_height = self.maze.shape[0]
        self.start_state = self.compress([1, 1])
        self.goal_state = self.compress([self.maze_height-2, self.maze_width-2])
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.state_space = DiscreteSpace(self.maze_width*self.maze_height)
        self.action_space = DiscreteSpace(len(self.actions))

    def compress(self, s):
        return self.maze_width * s[0] + s[1]

    def uncompress(self, s):
        y = int(s / self.maze_width)
        return [y, s - (self.maze_width * y)]

    def step(self, s, a):
        assert a >= 0 and a < len(self.actions)
        restored = self.uncompress(s)
        y = restored[0]
        x = restored[1]
        dy, dx = self.actions[a]
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x >= self.maze_width:
            # off grid
            new_x = x
        if new_y < 0 or new_y >= self.maze_height:
            # off grid
            new_y = y
        if self.maze[new_y, new_x] == 1:
            # hit wall
            new_y = y
            new_x = x
        new_state = self.compress([new_y, new_x])
        reward = -1
        done = (new_state == self.goal_state)
        return new_state, reward, done

    def reset(self):
        ...

    def get_start_state(self):
        return self.start_state

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space
