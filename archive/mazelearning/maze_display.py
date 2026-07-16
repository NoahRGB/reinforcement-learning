import pygame

import math

def show_agents(agents, env):
    WIDTH, HEIGHT = 1500, 600
    max_iterations = 10000
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
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            running = not event.type == pygame.QUIT
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    scale *= 1.05 
                else:
                    scale *= 0.95
                font = pygame.font.SysFont("Arial", math.floor(font_size * scale))

            if event.type == pygame.MOUSEBUTTONUP:
                ...
            # if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            #     panning = True
            #     mouse_x, mouse_y = event.pos
            #     saved_mouse_x = mouse_x - camera_offset_x
            #     saved_mouse_y = mouse_y - camera_offset_y 
            # elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            #     panning = False
            # elif event.type == pygame.MOUSEMOTION and panning:
            #     mouse_x, mouse_y = event.pos
            #     camera_offset_x = (mouse_x - saved_mouse_x)
            #     camera_offset_y = (mouse_y - saved_mouse_y)
    
        if (interactive and pygame.key.get_pressed()[pygame.K_SPACE]) or not interactive: 
            for agent in agents:
                if agent.completed_iterations < max_iterations:
                    agent.iteration_step()
                    if agent.done and agent.completed_iterations < max_iterations:
                        agent.reset_iteration()
    
        display_agents(camera_offset_x, camera_offset_y, scale, agents, screen, font, env)
    
        pygame.display.flip()

    pygame.quit()

def get_optimal_action(qtable, i, j, env):
    actions = ["up", "down", "right", "left"]
    legal_actions = env.get_legal([i, j])
    legal_q_values = qtable[i][j][legal_actions]
    if len(legal_q_values) > 0:
        return actions[legal_q_values.argmax()]

def draw_arrow(surface, colour, direction, x_pos, y_pos, cs):
    pts = [
        (0.20, 0.47),
        (0.58, 0.47),
        (0.58, 0.38),
        (0.82, 0.50),
        (0.58, 0.62),
        (0.58, 0.53),
        (0.20, 0.53),
    ]

    def rotate(x, y):
        if direction == "right":
            return x, y
        elif direction == "down":
            return 1 - y, x
        elif direction == "left":
            return 1 - x, 1 - y
        elif direction == "up":
            return y, 1-x

    poly = []
    for x, y in pts:
        xr, yr = rotate(x, y)
        poly.append((x_pos + xr * cs, y_pos + yr * cs))

    pygame.draw.polygon(surface, colour, poly)

def display_agents(camera_offset_x, camera_offset_y, scl, agents, screen, font, env, max_x=1000):
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

        screen.blit(font.render(f"{agent.title}", False, (0, 0, 0)), (xoff, yoff + env.pixel_height * scl))
        screen.blit(font.render(f"{agent}", False, (0, 0, 0)), (xoff, yoff + env.pixel_height * scl + scl * 25))
        screen.blit(font.render(f"episode: {agent.completed_iterations}", False, (0, 0, 0)), (xoff, yoff + scl * env.pixel_height + scl * 50))

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
                pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(x_pos, y_pos, cs, cs), width=math.ceil(1*scl))

                optimal_action = get_optimal_action(agent.qtable, i, j, env)
                if optimal_action is not None:
                    draw_arrow(screen, (50, 50, 200), optimal_action, x_pos, y_pos, cs)

                if env.maze[i][j] != 0:
                    pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(x_pos, y_pos, cs, cs))
