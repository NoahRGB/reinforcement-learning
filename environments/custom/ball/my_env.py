import pygame

from ball import Ball
from surface import Surface

import numpy as np

FPS = 60
WIDTH, HEIGHT = 500, 500

BALL_START_X = 50
BALL_FLOOR_OFFSET = 200
BALL_RAD = 20 
BALL_MASS = 1 

FLOOR_Y = HEIGHT * 0.95

GRAVITY = 0.5
THROW_STRENGTH = 10.0

throw_angle = np.radians(45.0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

ball = Ball(screen, np.array([BALL_START_X, FLOOR_Y - (BALL_RAD / 2) + 1 - BALL_FLOOR_OFFSET]), BALL_RAD, BALL_MASS)
floor = Surface(screen, np.array([0.0, FLOOR_Y]), np.array([WIDTH, FLOOR_Y]), 1.0)
roof = Surface(screen, np.array([0.0, 0.0]), np.array([WIDTH, 0.0]), 1.0)
left_wall = Surface(screen, np.array([0.0, 0.0]), np.array([0.0, HEIGHT]), 0.8)
right_wall = Surface(screen, np.array([WIDTH, 0.0]), np.array([WIDTH, HEIGHT]), 0.8)

goal = Surface(screen, np.array([450, 200]), np.array([450, 300]), 1.0, col=(0, 0, 0))

is_ball_released = False

while running:

    # events
    for event in pygame.event.get():
        running = not event.type == pygame.QUIT
    
    pressed_mouse = pygame.mouse.get_pressed()
    pressed_keys = pygame.key.get_pressed()

    if pressed_keys[pygame.K_SPACE]:
        ball.apply_force(np.array([1.0, 0.0]))
    if not is_ball_released and pressed_mouse[0]:
        is_ball_released = True
        throw_force = np.array([np.cos(throw_angle), np.sin(throw_angle)])
        ball.apply_force(throw_force * THROW_STRENGTH)

    # rendering
    screen.fill((255, 255, 255))

    if ball.surface_collision(goal):
        ball.col = (0, 255, 0)
    else:
        ball.col = (255, 0, 0)

    floor.display()
    goal.display()

    if is_ball_released:
        ball.apply_force(np.array([0.0, GRAVITY]))

    ball.update(floor, roof, left_wall, right_wall)
    ball.display()

    # maintenance
    pygame.display.flip()
    clock.tick(FPS)

