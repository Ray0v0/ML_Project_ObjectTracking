# 传统控制器，通过传入info来决策油门刹车转向
from controller.controller import Controller


class TraditionalController(Controller):
    # 由于每帧进行一次决策，控制器需要获取时钟来得知当前运行fps，确保在高fps和低fps下表现一致
    def register_display_manager(self, display_manager):
        self.display_manager = display_manager

    def get_fps(self):
        return self.display_manager.clock.get_fps()

    @staticmethod
    def is_traditional_controller():
        return True