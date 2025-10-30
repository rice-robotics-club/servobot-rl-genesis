import pygame
import sys

pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((100, 100))
pygame.display.set_caption("Controller Test")

print(f"Joysticks found: {pygame.joystick.get_count()}")

if pygame.joystick.get_count() > 0:
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Controller: {joy.get_name()}")
    print(f"Axes: {joy.get_numaxes()}")
    print("Move sticks now...\n")
    
    clock = pygame.time.Clock()
    for i in range(300):  # 5 seconds at 60fps
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.JOYAXISMOTION:
                print(f"Frame {i}: Axis {event.axis} = {event.value:.2f}")
        
        axes = [joy.get_axis(i) for i in range(joy.get_numaxes())]
        if any(abs(a) > 0.1 for a in axes):
            print(f"Frame {i} (polled): {[f'{a:.2f}' for a in axes]}", flush=True)
        
        screen.fill((0, 0, 0))
        pygame.display.flip()
        clock.tick(60)
else:
    print("No controller detected!")