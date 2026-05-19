import gymnasium as gym
from gymnasium import spaces

import pygame

import numpy as np

from environments.custom.ball.surface import Surface
from environments.custom.ball.ball import Ball

class BallEnv(gym.Env):
    def __init__(self, width=500, height=500, render_mode=None):
        
        self.width = width
        self.height = height
        self.render_mode = render_mode

        self.gravity = 0.5
        self.floor_y = self.height * 0.95
        self.steps = 0
        self.max_steps = 250

        self.ball = Ball(None, np.array([50.0, 50.0]), 20, 1)
        self.floor = Surface(None, np.array([0.0, self.floor_y]), np.array([self.width, self.floor_y]), 1.0)
        self.roof = Surface(None, np.array([0.0, 0.0]), np.array([self.width, 0.0]), 1.0)
        self.left_wall = Surface(None, np.array([0.0, 0.0]), np.array([0.0, self.height]), 0.8)
        self.right_wall = Surface(None, np.array([self.width, 0.0]), np.array([self.width, self.height]), 0.8)

        self.goal = Surface(None, np.array([450, 200]), np.array([450, 300]), 1.0, col=(0, 0, 0))
        self.goal_midpoint = np.array([450, 250])

        self.is_ball_released = False

        self.observation_space = spaces.Box(np.array([0.0, 0.0], dtype=np.float32), 
                                            np.array([self.width, self.height], dtype=np.float32), 
                                            shape=(2,), 
                                            dtype=np.float32)
        
        self.action_space = spaces.Box(np.array([0.0, 0.0], dtype=np.float32), np.array([7.0, 180.0], dtype=np.float32), dtype=np.float32)
        
        self.window = None
        self.clock = None

    def render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.ball.screen = self.window
            self.floor.screen = self.window
            self.roof.screen = self.window
            self.left_wall.screen = self.window
            self.right_wall.screen = self.window
            self.goal.screen = self.window

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        self.floor.display()
        self.goal.display()
        self.ball.display()

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball = Ball(None if self.window is None else self.window, np.array([50.0, 50.0], dtype=np.float32), 20, 1)
        observation = self.ball.pos
        self.steps = 0
        info = {}

        if self.render_mode == "human":
            self.render_frame()

        return observation, info
    
    def calc_reward(self, hitting_goal):
        dist_to_goal_midpoint = np.sqrt((self.ball.pos[0] - self.goal_midpoint[0])**2 + (self.ball.pos[1] - self.goal_midpoint[1])**2)
        reward = -(dist_to_goal_midpoint / 500)
        if hitting_goal: reward += 30
        return reward
        
    def step(self, action):
        self.steps += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        throw_strength, throw_angle = action

        sim_done = False
        sim_steps = 0
        reward = 0
        sim_ball_pushed = False
        self.ball = Ball(None if self.window is None else self.window, np.array([50.0, 50.0], dtype=np.float32), 20, 1)
        while not sim_done:
            if not sim_ball_pushed: 
                sim_ball_pushed = True
                throw_angle = np.deg2rad(throw_angle)
                throw_force = np.array([np.cos(throw_angle), np.sin(throw_angle)]) * throw_strength
                self.ball.apply_force(throw_force)

            self.ball.apply_force(np.array([0.0, self.gravity]))
            self.ball.update(self.floor, self.roof, self.left_wall, self.right_wall)

            if self.render_mode == "human":
                self.render_frame()
            sim_steps += 1
            hitting_goal = self.ball.surface_collision(self.goal)
            if sim_steps >= self.max_steps or hitting_goal:
                sim_done = True
                reward = self.calc_reward(hitting_goal)

        self.ball = Ball(None if self.window is None else self.window, np.array([50.0, 50.0], dtype=np.float32), 20, 1)

        return self.ball.pos, reward, False, False, {}

    def close(self):
        if self.window is not None:
            pygame.quit()