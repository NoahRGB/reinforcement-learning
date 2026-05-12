import pygame

import numpy as np

class Ball:

    def __init__(self, screen, pos, radius, mass, col=(255, 0, 0)):
        self.screen = screen
        self.pos = pos
        self.vel = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])
        self.radius = radius
        self.mass = mass
        self.col = col
    
    def surface_collision(self, surface):
        d = surface.end_pos - surface.start_pos
        f = surface.start_pos - self.pos
        a = d.dot(d)
        b = 2 * f.dot(d)
        c = f.dot(f) - self.radius*self.radius
        disc = b * b - 4 * a * c
        if disc < 0: return False
        disc = np.sqrt(disc)
        t1 = (-b - disc) / (2 * a)
        t2 = (-b + disc) / (2 * a)
        if t1 >= 0 and t1 <= 1: return True
        if t2 >= 0 and t2 <= 1: return True
        return False

    def edges(self, floor, roof, left_wall, right_wall):
        if (self.pos[1] + self.radius) > floor.start_pos[1]:
            self.vel[1] *= -floor.bounciness
            self.pos[1] = floor.start_pos[1] - self.radius

        if (self.pos[1] - self.radius) < roof.start_pos[1]:
            self.vel[1] *= -roof.bounciness
            self.pos[1] = roof.start_pos[1] + self.radius

        if (self.pos[0] - self.radius) < (left_wall.start_pos[0]):
            self.vel[0] *= -left_wall.bounciness
            self.pos[0] = left_wall.start_pos[0] + self.radius

        if (self.pos[0] + self.radius) > (right_wall.start_pos[0]):
            self.vel[0] *= -right_wall.bounciness
            self.pos[0] = right_wall.start_pos[0] - self.radius

    def apply_force(self, force):
        self.acc += (force / self.mass)

    def update(self, floor, roof, left_wall, right_wall):
        self.edges(floor, roof, left_wall, right_wall)

        self.vel += self.acc
        self.pos += self.vel
        self.acc *= 0

    def display(self):
        pygame.draw.circle(self.screen, self.col, self.pos.tolist(), self.radius)