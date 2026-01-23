from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    def __init__(self, actions):
        self.actions = actions

    @abstractmethod
    def step(self, action, state):
        pass


class MazeEnvironment(Environment):
    def __init__(self, cell_size=25):
        # self.maze = np.array([
        #     [1,1,1,1,1],
        #     [1,0,0,0,1],
        #     [1,0,0,0,1],
        #     [1,1,1,1,1],
        # 
        # ])
        # self.maze = np.array([
        #     [1,1,1,1,1,1,1,1,1],
        #     [1,0,1,0,0,0,0,0,1],
        #     [1,0,1,1,1,0,1,0,1],
        #     [1,0,0,0,1,0,1,0,1],
        #     [1,1,1,0,1,0,1,0,1],
        #     [1,0,0,0,1,0,1,0,1],
        #     [1,0,1,1,1,0,1,1,1],
        #     [1,0,1,0,0,0,0,0,1],
        #     [1,0,1,1,1,1,1,0,1],
        #     [1,0,0,0,0,0,0,0,1],
        #     [1,1,1,1,1,1,1,0,1],
        #     [1,0,0,0,0,0,1,0,1],
        #     [1,0,1,0,1,1,1,0,1],
        #     [1,0,1,0,0,0,1,0,1],
        #     [1,0,1,1,1,0,1,0,1],
        #     [1,0,0,0,1,0,0,0,1],
        #     [1,1,1,1,1,1,1,1,1]])
        super().__init__([[-1, 0], [1, 0], [0, -1], [0, 1]])
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
        self.cell_size = cell_size
        self.maze_width = self.maze.shape[1]
        self.maze_height = self.maze.shape[0]
        self.pixel_width = self.maze_width * self.cell_size
        self.pixel_height = self.maze_height * self.cell_size
        self.start_state = [1, 1]
        self.goal_state = [self.maze_height-2, self.maze_width-2]

    def get_legal(self, state):
        y, x = state
        legal_moves = []
        for move in self.actions:
            dy, dx = move
            new_x, new_y = x + dx, y + dy
            if (new_x > 0 and new_x < self.maze_width 
                and new_y > 0 and new_y < self.maze_height
                and self.maze[new_y, new_x] != 1):
                legal_moves.append(self.actions.index(move))
        return legal_moves

    def step(self, action, state):
        assert action >= 0 and action < len(self.actions)
        y, x = state
        dy, dx = self.actions[action]
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
        new_state = [new_y, new_x]
        reward = -1
        done = (new_state==self.goal_state)
        return new_state, reward, done
 
