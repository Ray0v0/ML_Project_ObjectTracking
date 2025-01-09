import pygame
import numpy as np

# pygame窗口控制器
class DisplayManager:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.font = self.__get_font__()
        self.clock = pygame.time.Clock()

    @staticmethod
    def __get_font__():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)

    @staticmethod
    def should_quit():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False

    # 显示画面
    def draw(self, image):
        array = np.frombuffer(image.raw_data, np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.display.blit(image_surface, (0, 0))

    def draw_box(self, box):
        pygame.draw.rect(self.display, (255, 0, 0), (box[0], box[1], box[2] - box[0], box[3] - box[1]), 2)

    # 显示帧率
    def write_fps(self, fps):
        self.display.blit(self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),(8, 10))
        self.display.blit(self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)), (8, 28))

    # 刷新画面
    def flip(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()