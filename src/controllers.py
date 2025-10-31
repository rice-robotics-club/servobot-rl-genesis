import pygame

class Controller:
    def __init__(self, type="keyboard"):
        self.joystick = None
        self.ps4 = (type == "ps4")
        self.xbox = (type == "xbox")
        self.keyboard = (type == "keyboard")
        self.target_speed = 0.5  # m/s
        self.screen = None
        self.clock = None
        if not (self.ps4 or self.xbox or self.keyboard):
            raise ValueError("Controller type must be 'ps4', 'xbox', or 'keyboard'")

    def initialize(self):
        """Initialize the pygame input system and joystick."""
        if self.xbox:
            print("Xbox controller selected.")
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.screen = pygame.display.set_mode((100, 100))
                pygame.display.set_caption("Controller Input... Keep in Focus!")
                self.clock = pygame.time.Clock()

                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Initialized joystick: {self.joystick.get_name()}")
                print("NOTE: Keep the pygame window in focus for controller input to work!")
            else:
                print("No Xbox controller detected!")
        elif self.ps4:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Initialized joystick: {self.joystick.get_name()}")
            else:
                print("No joystick detected!")
        elif self.keyboard:
            pygame.init()
            self.screen = pygame.display.set_mode((100, 100))
            self.clock = pygame.time.Clock()
            print("Keyboard control selected. Use arrow keys for movement and [ ] for rotation.")

    def get_command(self):
        """Get the current command from the controller."""
        command = (0.0, 0.0, 0.0)
        if self.xbox:
            # Check for quit condition to keep pygame responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return (0.0, 0.0, 0.0)

            # Xbox controller standard mapping:
            # Axis 0: Left stick X (strafe left/right)
            # Axis 1: Left stick Y (forward/backward)
            # Axis 2: Right stick X (rotation)
            # Axis 4: LT (Left Trigger) - returns -1.0 (not pressed) to 1.0 (fully pressed)
            # Axis 5: RT (Right Trigger) - returns -1.0 (not pressed) to 1.0 (fully pressed)
            forward = -self.joystick.get_axis(1)  # Left stick Y
            strafe = self.joystick.get_axis(0)     # Left stick X
            rotate = self.joystick.get_axis(2)     # Right stick X
            
            # Get trigger values (normalize from -1..1 to 0..1)
            lt_value = (self.joystick.get_axis(4) + 1.0) / 2.0  # Left trigger
            rt_value = (self.joystick.get_axis(5) + 1.0) / 2.0  # Right trigger
            
            throttle = rt_value - lt_value  # RT speeds up, LT slows down

            self.target_speed += throttle * 0.01  # Adjust speed
            self.target_speed = max(0.1, min(5.0, self.target_speed))  # Clamp between 0.1 and 5.0

            str_forward = max(0.0, min(forward, 1.0))
            str_back = max(0.0, min(-forward, 1.0))
            str_left = max(0.0, min(-strafe, 1.0))
            str_right = max(0.0, min(strafe, 1.0))
            rot_left = max(0.0, min(-rotate, 1.0))
            rot_right = max(0.0, min(rotate, 1.0))


            command = (forward * self.target_speed, strafe * self.target_speed, rotate * self.target_speed * 4)
            self.screen.fill((0, 0, 0))
            self.draw_movement_icons(str_forward, str_back, str_left, str_right, rot_left, rot_right)
            pygame.display.flip()
            self.clock.tick(60)

        elif self.ps4:
            for _ in pygame.event.get():
                pass

            command = (
                -self.joystick.get_axis(2),
                self.joystick.get_axis(3),
                -self.joystick.get_axis(0)
            ) if self.joystick else None

        
        elif self.keyboard:
            # Track key state ourselves instead of trusting pygame
            if not hasattr(self, 'keys_down'):
                self.keys_down = set()
            
            forward = 0.0
            strafe = 0.0
            rotate = 0.0
            
            # Track key states
            str_forward = 0.0
            str_back = 0.0
            str_left = 0.0
            str_right = 0.0
            rot_left = 0.0
            rot_right = 0.0

            # Process events and track key state manually
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return (0.0, 0.0, 0.0)
                if event.type == pygame.KEYDOWN:
                    self.keys_down.add(event.key)
                    print(f"KEYDOWN: {pygame.key.name(event.key)}")
                if event.type == pygame.KEYUP:
                    self.keys_down.discard(event.key)
                    print(f"KEYUP: {pygame.key.name(event.key)}")
            
            # Use our tracked state instead of get_pressed()
            if pygame.K_UP in self.keys_down:
                str_forward = 1.0
                forward = 1.0
            if pygame.K_DOWN in self.keys_down:
                str_back = 1.0
                forward = -1.0
            if pygame.K_LEFT in self.keys_down:
                str_left = 1.0
                strafe = -1.0
            if pygame.K_RIGHT in self.keys_down:
                str_right = 1.0
                strafe = 1.0
            if pygame.K_LEFTBRACKET in self.keys_down:
                rot_left = 1.0
                rotate = -1.0
            if pygame.K_RIGHTBRACKET in self.keys_down:
                rot_right = 1.0
                rotate = 1.0
            
            # Debug: print final command
            if forward != 0.0 or strafe != 0.0 or rotate != 0.0:
                print(f"Command: forward={forward}, strafe={strafe}, rotate={rotate}")
            
            command = (forward * 0.5, strafe * 0.5, rotate * 0.5)
            self.screen.fill((200, 200, 200))
            self.draw_movement_icons(str_forward, str_back, str_left, str_right, rot_left, rot_right)
            pygame.display.flip()
            self.clock.tick(60)
        
        return command

    def draw_movement_icons(self, str_forward: float, str_back: float, str_left: float, str_right: float, rot_left: float, rot_right: float):
            
            # Draw icons on the pygame window for the directions 
            # and fill them in depending on which keys are pressed.
            icon_size = 15
            icon_gap = 12
            center_x = 50
            center_y = 40
            # Up arrow
            up_color = ((1 - str_forward) * 255, 255, (1 - str_forward) * 255)
            pygame.draw.polygon(self.screen, up_color, [
                (center_x, center_y - icon_size - icon_gap),
                (center_x - icon_size // 2, center_y - icon_gap),
                (center_x + icon_size // 2, center_y - icon_gap)
            ])
            # Down arrow
            down_color = ((1 - str_back) * 255, 255, (1 - str_back) * 255)
            pygame.draw.polygon(self.screen, down_color, [
                (center_x, center_y + icon_size + icon_gap),
                (center_x - icon_size // 2, center_y + icon_gap),
                (center_x + icon_size // 2, center_y + icon_gap)
            ])
            # Left arrow
            left_color = ((1 - str_left) * 255, 255, (1 - str_left) * 255)
            pygame.draw.polygon(self.screen, left_color, [
                (center_x - icon_size - icon_gap, center_y),
                (center_x - icon_gap, center_y - icon_size // 2),
                (center_x - icon_gap, center_y + icon_size // 2)
            ])
            # Right arrow
            right_color = ((1 - str_right) * 255, 255, (1 - str_right) * 255)
            pygame.draw.polygon(self.screen, right_color, [
                (center_x + icon_size + icon_gap, center_y),
                (center_x + icon_gap, center_y - icon_size // 2),
                (center_x + icon_gap, center_y + icon_size // 2)
            ])
            # Rotate left bracket
            rot_left_color = ((1 - rot_left) * 255, 255, (1 - rot_left) * 255)
            pygame.draw.polygon(self.screen, rot_left_color, [
                (center_x - icon_size - 2 * icon_gap, center_y - icon_size - icon_gap),
                (center_x - icon_size - 2 * icon_gap, center_y + icon_size + icon_gap),
                (center_x - icon_size - 2 * icon_gap + 5, center_y + icon_size + icon_gap - 5),
                (center_x - icon_size - 2 * icon_gap + 5, center_y - icon_size - icon_gap + 5)
            ])
            # Rotate right bracket
            rot_right_color = ((1 - rot_right) * 255, 255, (1 - rot_right) * 255)
            pygame.draw.polygon(self.screen, rot_right_color, [
                (center_x + icon_size + 2 * icon_gap, center_y - icon_size - icon_gap),
                (center_x + icon_size + 2 * icon_gap, center_y + icon_size + icon_gap),
                (center_x + icon_size + 2 * icon_gap - 5, center_y + icon_size + icon_gap - 5),
                (center_x + icon_size + 2 * icon_gap - 5, center_y - icon_size - icon_gap + 5)
            ])
            # Display target speed at the bottom in a cool font
            font = pygame.font.SysFont(None, 20)
            speed_text = font.render(f"speed: {self.target_speed:.2f}", True, (255, 255, 255))
            self.screen.blit(speed_text, (15, 80))