import pygame

class Surface:
    def __init__(self, screen, start_pos, end_pos, bounciness, col=(0, 0, 0)):
        self.screen = screen
        self.start_pos = start_pos 
        self.end_pos = end_pos
        self.thickness = 1 
        self.bounciness = bounciness
        self.col = col

    def display(self):
        pygame.draw.line(self.screen, self.col, self.start_pos, self.end_pos, self.thickness)

