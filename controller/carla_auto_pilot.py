from controller.special_controller import SpecialController


class CarlaAutoPilot(SpecialController):
    @staticmethod
    def is_traditional_controller():
        return False