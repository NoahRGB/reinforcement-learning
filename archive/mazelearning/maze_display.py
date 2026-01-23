import pygame

import math

def show_agents(agents):
    WIDTH, HEIGHT = 1500, 600
    max_iterations = 300 
    running = True
    interactive = False 
    panning = False
    saved_mouse_x, saved_mouse_y = 0, 0
    camera_offset_x, camera_offset_y = 0, 0
    scale = 1.0
    font_size = 25
    
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", font_size)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze learning")

    while running:
        screen.fill((40, 40, 40))
        for event in pygame.event.get():
            running = not event.type == pygame.QUIT
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    scale *= 1.05 
                else:
                    scale *= 0.95
                font = pygame.font.SysFont("Arial", math.floor(font_size * scale))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                panning = True
                mouse_x, mouse_y = event.pos
                saved_mouse_x = mouse_x - camera_offset_x
                saved_mouse_y = mouse_y - camera_offset_y 
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                panning = False
            elif event.type == pygame.MOUSEMOTION and panning:
                mouse_x, mouse_y = event.pos
                camera_offset_x = (mouse_x - saved_mouse_x)
                camera_offset_y = (mouse_y - saved_mouse_y)
    
        if (interactive and pygame.key.get_pressed()[pygame.K_SPACE]) or not interactive: 
            for agent in agents:
                if agent.completed_iterations < max_iterations:
                    agent.iteration_step()
                    if agent.done and agent.completed_iterations < max_iterations:
                        agent.reset_iteration()
    
        display_agents(camera_offset_x, camera_offset_y, scale, agents, screen, font)
    
        pygame.display.flip()

    pygame.quit()


def display_agents(camera_offset_x, camera_offset_y, scl, agents, screen, font, max_x=1000):
    spacing = 30 * scl
    xoff, yoff = (camera_offset_x + spacing), (camera_offset_y + spacing)
    for agent in agents:
        env = agent.environment
        cs = env.cell_size * scl
        if agent != agents[0]:
            xoff += env.pixel_width * scl + spacing
            if xoff - camera_offset_x > max_x * scl:
                xoff = (camera_offset_x + spacing)
                yoff += 500 * scl

        screen.blit(font.render(f"{agent.title}", False, (255, 0, 0)), (xoff, yoff + env.pixel_height * scl))
        screen.blit(font.render(f"{agent}", False, (255, 0, 0)), (xoff, yoff + env.pixel_height * scl + scl * 25))
        screen.blit(font.render(f"episode: {agent.completed_iterations}", False, (0, 255, 0)), (xoff, yoff + scl * env.pixel_height + scl * 50))

        for state in agent.current_iteration_path:
            state_y, state_x, col = state
            pygame.draw.rect(screen, col, pygame.Rect(xoff + state_x * cs, yoff + state_y * cs, cs-(0.5 * scl), cs-(1 * scl)))

        start_state_y, start_state_x = env.start_state
        goal_state_y, goal_state_x = env.goal_state
        current_state_y, current_state_x = agent.state
        pygame.draw.rect(screen, (0, 225, 0), pygame.Rect(xoff + start_state_x * cs, yoff + start_state_y * cs, cs, cs))
        pygame.draw.rect(screen, (0, 225, 0), pygame.Rect(xoff + goal_state_x * cs, yoff + goal_state_y * cs, cs, cs))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(xoff + current_state_x * cs, yoff + current_state_y * cs, cs, cs))

        for i in range(0, len(env.maze)):
            for j in range(0, len(env.maze[i])):
                x_pos = xoff + j * cs
                y_pos = yoff + i * cs
                pygame.draw.rect(screen, (235, 235, 235), pygame.Rect(x_pos, y_pos, cs, cs), width=math.ceil(1*scl))
                if env.maze[i][j] != 0:
                    pygame.draw.rect(screen, (235, 235, 235), pygame.Rect(x_pos, y_pos, cs, cs))
