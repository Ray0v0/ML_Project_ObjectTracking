from controller.special_controller import SpecialController

# 循迹控制器，framework中识别到前车使用该控制器会从file中加载坐标并循迹
class PathFollower(SpecialController):
    def __init__(self, file):
        self.file = file
