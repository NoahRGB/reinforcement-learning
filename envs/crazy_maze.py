import gymnasium as gym
import numpy as np
import pygame

class CrazyMaze(gym.Env):
    def __init__(self, render_mode=None):

        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)

#         self.maze = """
# ###################################
# #S  #   #   #   #   #   #   #   #2#
# # # # # # # # # # # # # # # # # # #
# #1# # # # # # # # # # # # # # # # #
# ### # # # # # # # # # # # # # # # #
# ### # # # # # # # # # # # # # # # #
# ### # # # # # # # # # # # # # # # #
# ### # # # # # # # # # # # # # # # #
# ###   #   #   #   #   #   #   #   #
# ###################################"""

#         self.maze = """
# ###########
# #S  #   #2#
# # # # # # #
# #1# # # # #
# ### # # # #
# ### # # # #
# ### # # # #
# ### # # # #
# ###   #   #
# ###########"""

        self.maze = """
###############
#S  #   #   #2#
# # # # # # # #
#1# # # # # # #
### # # # # # #
### # # # # # #
### # # # # # #
### # # # # # #
###   #   #   #
###############"""

#         self.maze = """
# ###########
# #S  #   #2#
# # # # # # #
# # # # # # #
# # # # # # #
# # # # # # #
# # # # # # #
# # # # # # #
# #1#   #   #
# ###########"""
        
        self.rows = [row for row in self.maze.split("\n") if row != ""]
        equal = [row for row in self.rows if len(row) == len(self.rows[0])]
        assert len(self.rows) == len(equal), "All rows must be same length"

        self.num_rows = len(self.rows)
        self.num_cols = len(self.rows[0])
        self.player_row_idx, self.player_col_idx = 1, 1

        self.screen_width, self.screen_height = 1000, 500
        self.cell_width = self.screen_width // self.num_cols
        self.cell_height = self.screen_height // self.num_rows

        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=self.num_rows*self.num_cols,
        #     dtype=np.int64,
        # )

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.num_rows - 1, self.num_cols - 1]),
            dtype=np.int64,
        )

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()


    def _render_frame(self):
        self.screen.fill((0, 0, 0))
        for row_idx in range(self.num_rows):
            for col_idx in range(self.num_cols):

                current_cell = self.rows[row_idx][col_idx]

                width = 1 if current_cell == " " else 0
                colour = (255, 0, 0)
                if current_cell == "#" or current_cell == " ":
                    colour = (255, 255, 255)
                if current_cell == "1":
                    colour = (0, 105, 16)
                if current_cell == "2":
                    colour = (0, 255, 0)

                pygame.draw.rect(
                    self.screen, 
                    colour, 
                    (col_idx * self.cell_width, row_idx * self.cell_height, self.cell_width, self.cell_height),
                    width=width
                )
        
        pygame.draw.rect(self.screen, (198, 219, 7), 
                         (self.player_col_idx * self.cell_width+(self.cell_width/4), self.player_row_idx * self.cell_height +(self.cell_height/4), self.cell_width*0.5, self.cell_height*0.5),
                         width=0)
        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_row_idx, self.player_col_idx = 1, 1

        # obs = np.array([self.player_row_idx * self.num_cols + self.player_col_idx])
        obs = np.array([self.player_row_idx, self.player_col_idx])
        return obs, {}

    def step(self, action):
        # 0 is move up
        # 1 is move right
        # 2 is move down
        # 3 is move left
        row_incs = [-1, 0, 1, 0]
        col_incs = [0, 1, 0, -1]

        new_player_row_idx = self.player_row_idx + row_incs[action]
        new_player_col_idx = self.player_col_idx + col_incs[action]

        reward = -1.0
        terminated = False
        truncated = False

        if new_player_row_idx >= 0 and new_player_row_idx < self.num_rows and new_player_col_idx >= 0 and new_player_col_idx < self.num_cols:

            new_cell = self.rows[new_player_row_idx][new_player_col_idx]
            if not new_cell == "#":
                self.player_row_idx = new_player_row_idx
                self.player_col_idx = new_player_col_idx

            if new_cell == "1":
                reward = 5
                terminated = True
            elif new_cell == "2":
                reward = 100000
                terminated = True

        # obs = np.array([self.player_row_idx * self.num_cols + self.player_col_idx])
        obs = np.array([self.player_row_idx, self.player_col_idx])

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {}

    def render(self):
        ...

    def close(self):
        if self.render_mode == "human":
            pygame.quit()