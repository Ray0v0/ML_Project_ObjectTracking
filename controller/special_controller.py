# 非传统控制器，目前来说有CarlaAutoPilot和PathFollower
# framework在识别到非传统控制器时需要进行特殊处理
from controller.controller import Controller


class SpecialController(Controller):
    @staticmethod
    def is_traditional_controller():
        return False