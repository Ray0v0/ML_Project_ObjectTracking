from controller.special_controller import SpecialController


class PathFollower(SpecialController):
    def __init__(self, file):
        self.file = file
