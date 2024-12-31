from controller.special_controller import SpecialController

# 自动驾驶控制器，framework中识别到前车使用该控制器就会打开auto_pilot()
class CarlaAutoPilot(SpecialController):
    @staticmethod
    def is_traditional_controller():
        return False